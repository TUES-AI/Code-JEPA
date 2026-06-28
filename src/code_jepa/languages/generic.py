"""Generic tree-sitter/text language adapter used by non-Python CodeSearchNet languages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from code_jepa.data.python_ast import line_count, rough_token_len
from code_jepa.languages.base import (
    CANONICAL_TRANSFORM_INVENTORY,
    LanguageUnit,
    ParseStatus,
    TRANSFORM_STAGES,
    role_transform_names,
)
from code_jepa.transforms.python_ast import TransformResult

try:  # pragma: no cover - dependency is exercised by dataset smoke scripts.
    from tree_sitter_language_pack import get_parser
except Exception:  # pragma: no cover
    get_parser = None


IDENT_RE = re.compile(r"\b[A-Za-z_$][A-Za-z0-9_$]*\b")
CALL_RE = re.compile(r"\b([A-Za-z_$][A-Za-z0-9_$.]*)\s*\(")
STRING_RE = re.compile(r"('(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")", re.S)


@dataclass(frozen=True)
class LanguageConfig:
    language: str
    parser_language: str
    line_comment: str = "//"
    block_comment_start: str = "/*"
    block_comment_end: str = "*/"
    keywords: frozenset[str] = frozenset()
    import_line_prefixes: tuple[str, ...] = ()
    statement_suffix: str = ";"
    bool_true: str = "true"
    bool_false: str = "false"
    supports_ternary: bool = True
    not_operator: str = "!"
    and_operator: str = "&&"
    or_operator: str = "||"


COMMON_KEYWORDS = frozenset(
    {
        "abstract",
        "and",
        "as",
        "async",
        "await",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "def",
        "defer",
        "do",
        "else",
        "elsif",
        "end",
        "enum",
        "except",
        "false",
        "finally",
        "for",
        "func",
        "function",
        "if",
        "import",
        "in",
        "interface",
        "let",
        "new",
        "nil",
        "none",
        "not",
        "null",
        "or",
        "package",
        "private",
        "protected",
        "public",
        "raise",
        "rescue",
        "return",
        "static",
        "struct",
        "switch",
        "this",
        "throw",
        "throws",
        "true",
        "try",
        "use",
        "using",
        "var",
        "void",
        "while",
        "with",
        "yield",
    }
)


class GenericTreeSitterAdapter:
    def __init__(self, config: LanguageConfig) -> None:
        self.config = config
        self.language = config.language
        self.implemented_transforms = {
            stage: {role: set(names) for role, names in roles.items()}
            for stage, roles in CANONICAL_TRANSFORM_INVENTORY.items()
        }
        self._parser: Any | None = None

    def parse(self, code: str) -> ParseStatus:
        if get_parser is None:
            return ParseStatus(None, balanced_text(code), balanced_text(code), "tree_sitter_language_pack_missing")
        try:
            parser = self._get_parser()
            tree = parser.parse(self._wrap_for_parse(code))
            root = tree.root_node()
            ok = not bool(root.has_error())
            return ParseStatus(tree, ok, ok, None if ok else "tree_sitter_error")
        except Exception as exc:  # pragma: no cover - defensive for parser package differences.
            ok = balanced_text(code)
            return ParseStatus(None, ok, ok, f"{type(exc).__name__}: {exc}")

    def _get_parser(self) -> Any:
        if self._parser is None:
            assert get_parser is not None
            self._parser = get_parser(self.config.parser_language)
        return self._parser

    def _wrap_for_parse(self, code: str) -> str:
        code = code.rstrip() + "\n"
        if self.language == "java":
            if re.search(r"\b(class|interface|enum|record)\b", code):
                return code
            return "class __CodeJepaWrapper {\n" + code + "\n}\n"
        if self.language == "go":
            return code if code.lstrip().startswith("package ") else "package main\n" + code
        if self.language == "php":
            stripped = code.lstrip()
            prefix = "" if stripped.startswith("<?") else "<?php\n"
            body = stripped[5:].lstrip() if stripped.startswith("<?php") else code
            if re.match(r"(?:(?:public|private|protected|static|final|abstract)\s+)+function\b", body.lstrip()):
                return prefix + "class __CodeJepaWrapper {\n" + body + "\n}\n"
            return prefix + body
        return code

    def extract_units(self, file_id: str, source: str, cfg: Any) -> list[LanguageUnit]:
        source = source.rstrip() + "\n"
        parsed = self.parse(source)
        units: list[LanguageUnit] = []
        if parsed.tree is not None:
            units.extend(self._tree_units(source, parsed.tree, cfg.max_units_per_file))
        if not units:
            units.append(self._whole_source_unit(source))
        if len(units) < cfg.max_units_per_file:
            units.extend(self._file_windows(source, cfg, remaining=cfg.max_units_per_file - len(units)))
        out = []
        seen = set()
        for unit in units:
            key = unit.code.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(unit)
            if len(out) >= cfg.max_units_per_file:
                break
        return out

    def _tree_units(self, source: str, tree: Any, max_units: int) -> list[LanguageUnit]:
        root = tree.root_node()
        units: list[LanguageUnit] = []
        for node in walk_nodes(root):
            kind = node_kind(node)
            if kind not in function_node_kinds(self.language):
                continue
            start = int(node_start_byte(node))
            end = int(node_end_byte(node))
            if start < 0 or end <= start or end > len(source.encode("utf-8")):
                continue
            segment = byte_slice(source, start, end).strip() + "\n"
            if rough_token_len(segment) < 3:
                continue
            name = node_name(node, segment) or f"{kind}_{len(units)}"
            start_line = node_start_line(node)
            end_line = node_end_line(node)
            units.append(self._unit(segment, "function" if "function" in kind else "method", name, start_line, end_line))
            if len(units) >= max_units:
                return units
        return units

    def _whole_source_unit(self, source: str) -> LanguageUnit:
        return self._unit(source, "function", "codesearchnet_row", 1, line_count(source))

    def _unit(self, code: str, family: str, name: str, start_line: int, end_line: int) -> LanguageUnit:
        return LanguageUnit(
            unit_family=family,
            unit_type=family,
            qualified_name=name,
            code=code.rstrip() + "\n",
            start_line=start_line,
            end_line=end_line,
            imports_context=self.imports_context(code),
            identifiers=self.identifiers_from_code(code),
            calls=self.calls_from_code(code),
            ast_sequence=self.ast_sequence_from_code(code),
        )

    def _file_windows(self, source: str, cfg: Any, *, remaining: int) -> list[LanguageUnit]:
        lines = source.splitlines()
        if len(lines) < cfg.file_window_lines:
            return []
        out: list[LanguageUnit] = []
        for index, start in enumerate(range(0, len(lines), cfg.file_window_stride)):
            if len(out) >= min(remaining, cfg.max_file_windows):
                break
            end = min(len(lines), start + cfg.file_window_lines)
            code = "\n".join(lines[start:end]).rstrip() + "\n"
            out.append(self._unit(code, "file_window", f"file_window_{index:03d}", start + 1, end))
        return out

    def spans(self, unit_id: str, code: str, max_spans: int) -> list[dict[str, Any]]:
        spans = []
        for index, match in enumerate(IDENT_RE.finditer(code)):
            if len(spans) >= max_spans:
                break
            line, col = line_col_from_offset(code, match.start())
            end_line, end_col = line_col_from_offset(code, match.end())
            spans.append(
                {
                    "span_id": f"{unit_id}:tok:{index}",
                    "unit_id": unit_id,
                    "span_index": index,
                    "node_type": "identifier",
                    "start_line": line,
                    "start_col": col,
                    "end_line": end_line,
                    "end_col": end_col,
                    "start_byte": len(code[: match.start()].encode("utf-8")),
                    "end_byte": len(code[: match.end()].encode("utf-8")),
                    "source": match.group(0),
                }
            )
        return spans

    def positive_views_for_stage(
        self,
        code: str,
        stage: str,
        *,
        max_views: int = 16,
        only_transform: str = "",
    ) -> list[TransformResult]:
        return self._views(code, stage, "positive", max_views=max_views, only_transform=only_transform)

    def negative_views_for_stage(
        self,
        code: str,
        stage: str,
        *,
        max_views: int = 8,
        only_transform: str = "",
    ) -> list[TransformResult]:
        return self._views(code, stage, "negative", max_views=max_views, only_transform=only_transform)

    def _views(
        self,
        code: str,
        stage: str,
        role: str,
        *,
        max_views: int,
        only_transform: str,
    ) -> list[TransformResult]:
        out = []
        seen = {code.strip()}
        functions = positive_transform_functions(self) if role == "positive" else negative_transform_functions(self)
        for name in role_transform_names(stage, role, only_transform=only_transform):
            result = functions[name](code)
            if result is None:
                continue
            normalized = result.code.strip()
            if not normalized or normalized in seen:
                continue
            if not self.parse(result.code).parse_ok:
                continue
            seen.add(normalized)
            out.append(result)
            if len(out) >= max_views:
                break
        return out

    def imports_context(self, code: str) -> str:
        lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if any(stripped.startswith(prefix) for prefix in self.config.import_line_prefixes):
                lines.append(line)
                continue
            break
        return "\n".join(lines[:128])

    def identifiers_from_code(self, code: str) -> list[str]:
        return sorted({m.group(0) for m in IDENT_RE.finditer(strip_strings_and_comments(code, self.config)) if not self._is_keyword(m.group(0))})[:512]

    def calls_from_code(self, code: str) -> list[str]:
        calls = []
        for match in CALL_RE.finditer(strip_strings_and_comments(code, self.config)):
            name = match.group(1)
            base = name.split(".")[-1]
            if self._is_keyword(base):
                continue
            if name not in calls:
                calls.append(name)
        return calls[:512]

    def ast_sequence_from_code(self, code: str) -> list[str]:
        parsed = self.parse(code)
        if parsed.tree is None:
            return []
        out = []
        for node in walk_nodes(parsed.tree.root_node()):
            out.append(node_kind(node))
            if len(out) >= 1024:
                break
        return out

    def _is_keyword(self, name: str) -> bool:
        return name.lower() in COMMON_KEYWORDS or name in self.config.keywords


TransformFn = Callable[[str], TransformResult | None]


def positive_transform_functions(adapter: GenericTreeSitterAdapter) -> dict[str, TransformFn]:
    return {
        "surface_normalize": lambda code: surface_normalize(adapter, code),
        "remove_comments_or_docstrings": lambda code: remove_comments(adapter, code),
        "rename_locals": lambda code: rename_locals(adapter, code),
        "swap_independent_assignments": lambda code: swap_independent_assignments(adapter, code),
        "bool_return_simplify": lambda code: bool_return_simplify(adapter, code),
        "if_return_merge": lambda code: if_return_merge(adapter, code),
        "remove_unreachable_else": lambda code: remove_unreachable_else(adapter, code),
        "import_sort_same_block": lambda code: import_sort_same_block(adapter, code),
        "range_loop_to_while_or_equivalent": lambda code: range_loop_to_while(adapter, code),
        "append_loop_to_collection_literal": lambda code: append_loop_to_collection_literal(adapter, code),
        "accumulator_loop_to_builtin": lambda code: accumulator_loop_to_builtin(adapter, code),
        "demorgan_rewrite": lambda code: demorgan_rewrite(adapter, code),
        "swap_independent_statement_block": lambda code: swap_independent_statement_block(adapter, code),
    }


def negative_transform_functions(adapter: GenericTreeSitterAdapter) -> dict[str, TransformFn]:
    return {
        "flip_comparison": lambda code: flip_comparison(adapter, code),
        "flip_boolop": lambda code: flip_boolop(adapter, code),
        "swap_call_args": lambda code: swap_call_args(adapter, code),
        "wrong_variable": lambda code: wrong_variable(adapter, code),
        "flip_small_integer": lambda code: flip_small_integer(adapter, code),
        "flip_membership_or_identity": lambda code: flip_membership_or_identity(adapter, code),
        "negate_condition": lambda code: negate_condition(adapter, code),
        "flip_arithmetic_operator": lambda code: flip_arithmetic_operator(adapter, code),
        "flip_subscript_index": lambda code: flip_subscript_index(adapter, code),
        "flip_default_value": lambda code: flip_default_value(adapter, code),
        "flip_sort_reverse": lambda code: flip_sort_reverse(adapter, code),
        "remove_return_value": lambda code: remove_return_value(adapter, code),
        "remove_async_or_concurrency_wait": lambda code: remove_async_or_concurrency_wait(adapter, code),
        "shift_loop_bound": lambda code: shift_loop_bound(adapter, code),
        "remove_guard_branch": lambda code: remove_guard_branch(adapter, code),
        "flip_exception_type": lambda code: flip_exception_type(adapter, code),
        "drop_keyword_argument": lambda code: drop_keyword_argument(adapter, code),
        "copy_to_alias_mutation": lambda code: copy_to_alias_mutation(adapter, code),
        "drop_resource_context": lambda code: drop_resource_context(adapter, code),
    }


def result(adapter: GenericTreeSitterAdapter, name: str, role: str, old: str, new: str, confidence: str, metadata: dict[str, Any] | None = None) -> TransformResult | None:
    if old.strip() == new.strip():
        return None
    span = changed_span(old, new)
    span.update({"kind": name})
    meta = {"stage_family_name": name, "language": adapter.language}
    if metadata:
        meta.update(metadata)
    if role == "negative":
        meta.setdefault("negative_type", name)
    else:
        meta.setdefault("positive_type", name)
    return TransformResult(name=name, role=role, code=new.rstrip() + "\n", confidence=confidence, changed_spans=[span], metadata=meta)


def surface_normalize(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    lines = [line.rstrip() for line in code.rstrip().splitlines()]
    compact = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        compact.append(line)
        previous_blank = blank
    return result(adapter, "surface_normalize", "positive", code, "\n".join(compact) + "\n", "safe")


def remove_comments(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    new = strip_comments_preserve_newlines(code, adapter.config)
    return result(adapter, "remove_comments_or_docstrings", "positive", code, new, "likely")


def rename_locals(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    clean = strip_strings_and_comments(code, adapter.config)
    counts: dict[str, int] = {}
    order = []
    for match in IDENT_RE.finditer(clean):
        name = match.group(0)
        if adapter._is_keyword(name) or name[0].isupper():
            continue
        counts[name] = counts.get(name, 0) + 1
        if name not in order:
            order.append(name)
    target = next((name for name in order if counts.get(name, 0) >= 2), "")
    if not target:
        return None
    replacement = f"cj_{target}"
    new = re.sub(rf"\b{re.escape(target)}\b", replacement, code)
    return result(adapter, "rename_locals", "positive", code, new, "likely", {"renamed": target, "replacement": replacement})


def flip_comparison(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    return replace_operator(adapter, code, "flip_comparison", {"<=": "<", ">=": ">", "==": "!=", "!=": "==", "<": "<=", ">": ">="}, "comparison_operator")


def flip_boolop(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    mapping = {adapter.config.and_operator: adapter.config.or_operator, adapter.config.or_operator: adapter.config.and_operator}
    if adapter.config.and_operator != "and":
        mapping.update({"&&": "||", "||": "&&"})
    mapping.update({" and ": " or ", " or ": " and "})
    for old, new_op in sorted(mapping.items(), key=lambda item: -len(item[0])):
        if old in code:
            return result(adapter, "flip_boolop", "negative", code, code.replace(old, new_op, 1), "likely", {"negative_type": "boolean_operator"})
    return None


def swap_call_args(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    pattern = re.compile(r"\b(?!if\b|for\b|while\b|switch\b|catch\b|return\b)([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(\s*([^,(){};\n]+)\s*,\s*([^,(){};\n]+)(\s*(?:,|\)))")
    match = pattern.search(code)
    if not match:
        return None
    tail = match.group(4)
    repl_tail = tail if tail.strip().startswith(",") else ")"
    replacement = f"{match.group(1)}({match.group(3).strip()}, {match.group(2).strip()}{repl_tail}"
    new = code[: match.start()] + replacement + code[match.end() :]
    return result(adapter, "swap_call_args", "negative", code, new, "likely", {"negative_type": "swapped_args"})


def wrong_variable(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    names = adapter.identifiers_from_code(code)
    names = [name for name in names if len(name) > 1 and not name[0].isupper()]
    if len(names) < 2:
        return None
    source, replacement = names[0], names[1]
    matches = list(re.finditer(rf"\b{re.escape(source)}\b", code))
    if not matches:
        return None
    match = matches[min(1, len(matches) - 1)]
    new = code[: match.start()] + replacement + code[match.end() :]
    return result(adapter, "wrong_variable", "negative", code, new, "likely", {"negative_type": "wrong_variable", "source": source, "replacement": replacement})


def flip_small_integer(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    for match in re.finditer(r"(?<![\w.])(-?\d+)(?![\w.])", code):
        value = int(match.group(1))
        if -9 <= value <= 99:
            replacement = str(value + 1 if value <= 0 else value - 1)
            new = code[: match.start()] + replacement + code[match.end() :]
            return result(adapter, "flip_small_integer", "negative", code, new, "likely", {"negative_type": "small_integer"})
    return None


def swap_independent_assignments(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    lines = code.splitlines()
    for index in range(len(lines) - 1):
        first = simple_assignment(lines[index])
        second = simple_assignment(lines[index + 1])
        if not first or not second:
            continue
        lhs1, rhs1 = first
        lhs2, rhs2 = second
        if lhs1 == lhs2 or lhs1 in identifiers(rhs2, adapter) or lhs2 in identifiers(rhs1, adapter):
            continue
        new_lines = list(lines)
        new_lines[index], new_lines[index + 1] = new_lines[index + 1], new_lines[index]
        return result(adapter, "swap_independent_assignments", "positive", code, "\n".join(new_lines) + "\n", "likely")
    return None


def bool_return_simplify(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    t = adapter.config.bool_true
    f = adapter.config.bool_false
    patterns = [
        re.compile(rf"if\s*\(([^(){{}};\n]+)\)\s*\{{\s*return\s+({t}|{f})\s*;?\s*\}}\s*else\s*\{{\s*return\s+({t}|{f})\s*;?\s*\}}", re.S),
        re.compile(rf"if\s+([^{{}};\n]+)\s*\{{\s*return\s+({t}|{f})\s*\}}\s*else\s*\{{\s*return\s+({t}|{f})\s*\}}", re.S),
    ]
    match = next((pattern.search(code) for pattern in patterns if pattern.search(code)), None)
    if not match:
        return ruby_bool_return_simplify(adapter, code)
    cond, left, right = match.group(1).strip(), match.group(2), match.group(3)
    end = "" if adapter.language in {"go", "ruby"} else ";"
    if left == t and right == f:
        repl = f"return {cond}{end}"
    elif left == f and right == t:
        repl = f"return !({cond}){end}"
    else:
        return None
    return result(adapter, "bool_return_simplify", "positive", code, code[: match.start()] + repl + code[match.end() :], "safe")


def if_return_merge(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    patterns = [
        re.compile(r"if\s*\(([^(){};\n]+)\)\s*\{\s*return\s+([^;{}\n]+);?\s*\}\s*else\s*\{\s*return\s+([^;{}\n]+);?\s*\}", re.S),
        re.compile(r"if\s+([^{};\n]+)\s*\{\s*return\s+([^{}\n]+?)\s*\}\s*else\s*\{\s*return\s+([^{}\n]+?)\s*\}", re.S),
    ]
    match = next((pattern.search(code) for pattern in patterns if pattern.search(code)), None)
    if not match:
        return ruby_if_return_merge(adapter, code)
    cond, left, right = [item.strip().rstrip(";") for item in match.groups()]
    if adapter.config.supports_ternary:
        repl = f"return {cond} ? {left} : {right};"
    else:
        repl = f"if !({cond}) {{ return {right} }}\nreturn {left}"
    return result(adapter, "if_return_merge", "positive", code, code[: match.start()] + repl + code[match.end() :], "likely")


def remove_unreachable_else(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    patterns = [
        re.compile(r"if\s*\(([^(){};\n]+)\)\s*\{([^{}]*?return[^{}]*?;?)\s*\}\s*else\s*\{([^{}]+)\}", re.S),
        re.compile(r"if\s+([^{};\n]+)\s*\{([^{}]*?return[^{}]*?)\s*\}\s*else\s*\{([^{}]+)\}", re.S),
    ]
    match = next((pattern.search(code) for pattern in patterns if pattern.search(code)), None)
    if not match:
        return ruby_remove_unreachable_else(adapter, code)
    cond = match.group(1).strip()
    head = f"if {cond}" if adapter.language == "go" else f"if ({cond})"
    repl = f"{head} {{{match.group(2)}}}\n{match.group(3).strip()}"
    return result(adapter, "remove_unreachable_else", "positive", code, code[: match.start()] + repl + code[match.end() :], "safe")


def import_sort_same_block(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    lines = code.splitlines()
    indices = [i for i, line in enumerate(lines) if any(line.strip().startswith(prefix) for prefix in adapter.config.import_line_prefixes)]
    if len(indices) < 2:
        return None
    # Only sort the first contiguous import/use/require block.
    start = indices[0]
    end = start
    while end < len(lines) and any(lines[end].strip().startswith(prefix) or not lines[end].strip() for prefix in adapter.config.import_line_prefixes):
        end += 1
    block = [line for line in lines[start:end] if line.strip()]
    if len(block) < 2:
        return None
    sorted_block = sorted(block, key=lambda item: item.strip())
    if block == sorted_block:
        return None
    new_lines = [*lines[:start], *sorted_block, *lines[end:]]
    return result(adapter, "import_sort_same_block", "positive", code, "\n".join(new_lines) + "\n", "likely")


def flip_membership_or_identity(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    patterns = [
        (r"\s+instanceof\s+", " !instanceof "),
        (r"===", "!=="),
        (r"!==", "==="),
        (r"\bis\s+not\b", " is "),
        (r"\bis\b", " is not "),
        (r"\bin\b", " not in "),
        (r"\bnot\s+in\b", " in "),
        (r"==\s*nil\b", "!= nil"),
        (r"!=\s*nil\b", "== nil"),
        (r"==\s*null\b", "!= null"),
        (r"!=\s*null\b", "== null"),
    ]
    for pattern, replacement in patterns:
        match = re.search(pattern, code)
        if match:
            new = code[: match.start()] + replacement + code[match.end() :]
            return result(adapter, "flip_membership_or_identity", "negative", code, new, "likely", {"negative_type": "membership_or_identity"})
    return None


def negate_condition(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    match = re.search(r"\b(if|while)\s*\(([^(){};\n]+)\)", code)
    if not match:
        return ruby_negate_condition(adapter, code)
    repl = f"{match.group(1)} (!({match.group(2).strip()}))"
    return result(adapter, "negate_condition", "negative", code, code[: match.start()] + repl + code[match.end() :], "likely", {"negative_type": "condition_negation"})


def flip_arithmetic_operator(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    clean = strip_strings_and_comments(code, adapter.config)
    for pattern, replacement in [(r"(?<![+])\+(?![+=])", "-"), (r"(?<![-])-((?![-=]))", "+"), (r"\*", "/"), (r"/(?!/|\*)", "*")]:
        match = re.search(pattern, clean)
        if match:
            op_start = match.start()
            op_end = op_start + 1
            new = code[:op_start] + replacement + code[op_end:]
            return result(adapter, "flip_arithmetic_operator", "negative", code, new, "likely", {"negative_type": "arithmetic_operator"})
    return None


def flip_subscript_index(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    match = re.search(r"\[\s*(\d+|[A-Za-z_$][\w$]*)\s*\]", code)
    if not match:
        return None
    index = match.group(1)
    replacement = "1" if index == "0" else "0" if index.isdigit() else f"{index} + 1"
    new = code[: match.start(1)] + replacement + code[match.end(1) :]
    return result(adapter, "flip_subscript_index", "negative", code, new, "likely", {"negative_type": "subscript_index"})


def flip_default_value(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    match = re.search(r"([,(]\s*[A-Za-z_$][\w$]*(?:\s*:\s*[^,)=]+)?\s*=\s*)(true|false|null|nil|\d+|'[^']*'|\"[^\"]*\")", code)
    if not match:
        return None
    value = match.group(2)
    replacement = {"true": "false", "false": "true", "null": "0", "nil": "0"}.get(value, "0" if not value.startswith(("'", '"')) else "''")
    if value.isdigit():
        replacement = str(int(value) + 1)
    new = code[: match.start(2)] + replacement + code[match.end(2) :]
    return result(adapter, "flip_default_value", "negative", code, new, "likely", {"negative_type": "default_value"})


def flip_sort_reverse(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    replacements = [
        (".sort()", ".sort().reverse()"),
        (".sort();", ".sort();\nCollections.reverse(items);"),
        ("sort.Strings(", "sort.Sort(sort.Reverse(sort.StringSlice("),
        ("sort_by", "sort_by.reverse"),
        ("rsort(", "sort("),
        ("sort(", "rsort("),
    ]
    for old, new_text in replacements:
        if old in code:
            return result(adapter, "flip_sort_reverse", "negative", code, code.replace(old, new_text, 1), "likely", {"negative_type": "sort_reverse"})
    match = re.search(r"(<|>)", code)
    if match and "sort" in code.lower():
        new = code[: match.start()] + (">" if match.group(1) == "<" else "<") + code[match.end() :]
        return result(adapter, "flip_sort_reverse", "negative", code, new, "likely", {"negative_type": "sort_reverse"})
    return None


def remove_return_value(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    match = re.search(r"\breturn\s+([^;\n]+);", code)
    if not match:
        match = re.search(r"\breturn\s+([^\n]+)", code)
    if not match:
        return None
    new = code[: match.start()] + "return;" + code[match.end() :]
    if adapter.language == "ruby":
        new = code[: match.start()] + "return" + code[match.end() :]
    return result(adapter, "remove_return_value", "negative", code, new, "likely", {"negative_type": "return_value_removed"})


def remove_async_or_concurrency_wait(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    patterns = [r"\bawait\s+", r"\bgo\s+", r"<-\s*", r"\.join\(\)", r"\.get\(\)", r"\.wait\(\)", r"\.value\b"]
    for pattern in patterns:
        match = re.search(pattern, code)
        if match:
            new = code[: match.start()] + code[match.end() :]
            return result(adapter, "remove_async_or_concurrency_wait", "negative", code, new, "likely", {"negative_type": "async_or_concurrency_wait_removed"})
    return None


def range_loop_to_while(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    match = re.search(r"for\s*\(\s*([^;]+);\s*([^;]+);\s*([^)]+)\)\s*\{", code)
    if match:
        init, cond, inc = [item.strip() for item in match.groups()]
        insert_at = find_matching_brace(code, match.end() - 1)
        if insert_at is None:
            return None
        body = code[match.end() : insert_at].rstrip()
        repl = f"{init};\nwhile ({cond}) {{{body}\n{inc};\n}}"
        return result(adapter, "range_loop_to_while_or_equivalent", "positive", code, code[: match.start()] + repl + code[insert_at + 1 :], "likely")
    if adapter.language == "go":
        match = re.search(r"for\s+([^;\n]+):=\s*([^;\n]+);\s*([^;\n]+);\s*([^\{\n]+)\s*\{", code)
        if match:
            init_name, init_value, cond, inc = [item.strip() for item in match.groups()]
            close = find_matching_brace(code, match.end() - 1)
            if close is not None:
                body = code[match.end() : close].rstrip()
                repl = f"{init_name} := {init_value}\nfor {cond} {{{body}\n{inc}\n}}"
                return result(adapter, "range_loop_to_while_or_equivalent", "positive", code, code[: match.start()] + repl + code[close + 1 :], "likely")
    return ruby_range_loop_to_while(adapter, code)


def append_loop_to_collection_literal(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    if adapter.language == "javascript":
        match = re.search(r"(?:const|let|var)\s+(\w+)\s*=\s*\[\];\s*for\s*\((?:const|let|var)\s+(\w+)\s+of\s+([^)]*)\)\s*\{\s*\1\.push\(([^;]+)\);\s*\}", code, re.S)
        if match:
            out, var, iterable, expr = match.groups()
            repl = f"const {out} = {iterable.strip()}.map(({var}) => {expr.strip()});"
            return result(adapter, "append_loop_to_collection_literal", "positive", code, code[: match.start()] + repl + code[match.end() :], "likely")
    if adapter.language == "java":
        match = re.search(r"(?:List<[^>]+>|ArrayList<[^>]*>)\s+(\w+)\s*=\s*new\s+ArrayList<[^>]*>\(\);\s*for\s*\([^:]+\s+(\w+)\s*:\s*([^)]*)\)\s*\{\s*\1\.add\(([^;]+)\);\s*\}", code, re.S)
        if match:
            out, var, iterable, expr = match.groups()
            repl = f"List<?> {out} = {iterable.strip()}.stream().map({var} -> {expr.strip()}).collect(java.util.stream.Collectors.toList());"
            return result(adapter, "append_loop_to_collection_literal", "positive", code, code[: match.start()] + repl + code[match.end() :], "likely")
    if adapter.language == "go":
        match = re.search(r"(\w+)\s*:=\s*\[\]([^{}\n]+)\{\}\s*\n\s*for\s+([^\n{]+)\s*:=\s*range\s+([^\n{]+)\s*\{\s*\n\s*\1\s*=\s*append\(\1,\s*([^\n]+)\)\s*\n\s*\}", code, re.S)
        if match:
            out, typ, iterator, iterable, expr = match.groups()
            repl = f"{out} := func() []{typ.strip()} {{\n    {out} := []{typ.strip()}{{}}\n    for {iterator.strip()} := range {iterable.strip()} {{\n        {out} = append({out}, {expr.strip()})\n    }}\n    return {out}\n}}()"
            return result(adapter, "append_loop_to_collection_literal", "positive", code, code[: match.start()] + repl + code[match.end() :], "likely")
    if adapter.language == "php":
        match = re.search(r"(\$\w+)\s*=\s*\[\]\s*;\s*foreach\s*\(([^)]+)\s+as\s+(\$\w+)\)\s*\{\s*\1\s*\[\]\s*=\s*([^;]+);\s*\}", code, re.S)
        if match:
            out, iterable, var, expr = match.groups()
            repl = f"{out} = array_map(function({var}) {{ return {expr.strip()}; }}, {iterable.strip()});"
            return result(adapter, "append_loop_to_collection_literal", "positive", code, code[: match.start()] + repl + code[match.end() :], "likely")
    if adapter.language == "ruby":
        match = re.search(r"(\w+)\s*=\s*\[\]\s*\n\s*([^\n]+)\.each\s+do\s+\|(\w+)\|\s*\n\s*\1\s*<<\s*([^\n]+)\n\s*end", code)
        if match:
            out, iterable, var, expr = match.groups()
            repl = f"{out} = {iterable}.map {{ |{var}| {expr.strip()} }}"
            return result(adapter, "append_loop_to_collection_literal", "positive", code, code[: match.start()] + repl + code[match.end() :], "likely")
    return None


def accumulator_loop_to_builtin(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    if adapter.language == "javascript":
        match = re.search(r"(?:let|var)\s+(\w+)\s*=\s*0;\s*for\s*\((?:const|let|var)\s+(\w+)\s+of\s+([^)]*)\)\s*\{\s*\1\s*\+=\s*([^;]+);\s*\}", code, re.S)
        if match:
            total, var, iterable, expr = match.groups()
            repl = f"let {total} = {iterable.strip()}.reduce((acc, {var}) => acc + ({expr.strip()}), 0);"
            return result(adapter, "accumulator_loop_to_builtin", "positive", code, code[: match.start()] + repl + code[match.end() :], "likely")
    if adapter.language == "ruby":
        match = re.search(r"(\w+)\s*=\s*0\s*\n\s*([^\n]+)\.each\s+do\s+\|(\w+)\|\s*\n\s*\1\s*\+=\s*([^\n]+)\s*\n\s*end", code)
        if match:
            total, iterable, var, expr = match.groups()
            repl = f"{total} = {iterable}.sum {{ |{var}| {expr.strip()} }}"
            return result(adapter, "accumulator_loop_to_builtin", "positive", code, code[: match.start()] + repl + code[match.end() :], "likely")
    if adapter.language == "php":
        match = re.search(r"(\$\w+)\s*=\s*0\s*;\s*foreach\s*\(([^)]+)\s+as\s+(\$\w+)\)\s*\{\s*\1\s*\+=\s*([^;]+);\s*\}", code, re.S)
        if match:
            total, iterable, var, expr = match.groups()
            repl = f"{total} = array_sum(array_map(function({var}) {{ return {expr.strip()}; }}, {iterable.strip()}));"
            return result(adapter, "accumulator_loop_to_builtin", "positive", code, code[: match.start()] + repl + code[match.end() :], "likely")
    if adapter.language == "go":
        match = re.search(r"(\w+)\s*:=\s*0\s*\n\s*for\s+([^\n{]+)\s*:=\s*range\s+([^\n{]+)\s*\{\s*\n\s*\1\s*\+=\s*([^\n]+)\s*\n\s*\}", code, re.S)
        if match:
            total, iterator, iterable, expr = match.groups()
            repl = f"{total} := func() int {{\n    {total} := 0\n    for {iterator.strip()} := range {iterable.strip()} {{\n        {total} += {expr.strip()}\n    }}\n    return {total}\n}}()"
            return result(adapter, "accumulator_loop_to_builtin", "positive", code, code[: match.start()] + repl + code[match.end() :], "likely")
    if adapter.language == "java":
        match = re.search(r"int\s+(\w+)\s*=\s*0;\s*for\s*\([^:]+:\s*([^)]*)\)\s*\{\s*\1\s*\+=\s*([^;]+);\s*\}", code, re.S)
        if match:
            total, iterable, _expr = match.groups()
            repl = f"int {total} = {iterable.strip()}.stream().mapToInt(Integer::intValue).sum();"
            return result(adapter, "accumulator_loop_to_builtin", "positive", code, code[: match.start()] + repl + code[match.end() :], "likely")
    return None


def demorgan_rewrite(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    pattern = re.compile(r"!\s*\(\s*([^()&|]+?)\s*(&&|\|\|)\s*([^()&|]+?)\s*\)")
    match = pattern.search(code)
    if match:
        left, op, right = [item.strip() for item in match.groups()]
        new_op = "||" if op == "&&" else "&&"
        repl = f"!({left}) {new_op} !({right})"
        return result(adapter, "demorgan_rewrite", "positive", code, code[: match.start()] + repl + code[match.end() :], "safe")
    return ruby_demorgan_rewrite(adapter, code)


def swap_independent_statement_block(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    lines = code.splitlines()
    for index in range(len(lines) - 1):
        a, b = lines[index].strip(), lines[index + 1].strip()
        if not a or not b or any(token in a + b for token in ["return", "throw", "break", "continue", "if", "for", "while"]):
            continue
        if not (a.endswith(adapter.config.statement_suffix) or adapter.language == "ruby"):
            continue
        if not (b.endswith(adapter.config.statement_suffix) or adapter.language == "ruby"):
            continue
        new_lines = list(lines)
        new_lines[index], new_lines[index + 1] = new_lines[index + 1], new_lines[index]
        return result(adapter, "swap_independent_statement_block", "positive", code, "\n".join(new_lines) + "\n", "likely")
    return None


def shift_loop_bound(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    match = re.search(r"([A-Za-z_$][\w$]*)\s*(<|<=|>|>=)\s*([A-Za-z_$][\w$]*|\d+)", code)
    if not match:
        return None
    op = match.group(2)
    replacement = {"<": "<=", "<=": "<", ">": ">=", ">=": ">"}[op]
    new = code[: match.start(2)] + replacement + code[match.end(2) :]
    return result(adapter, "shift_loop_bound", "negative", code, new, "likely", {"negative_type": "loop_bound_off_by_one"})


def remove_guard_branch(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    match = re.search(r"\bif\s*\([^)]*\)\s*\{\s*(?:return|throw|raise)[^{};\n]*(?:;)?\s*\}\s*", code, re.S)
    if not match:
        return ruby_remove_guard_branch(adapter, code)
    new = code[: match.start()] + code[match.end() :]
    return result(adapter, "remove_guard_branch", "negative", code, new, "likely", {"negative_type": "missing_edge_case_branch"})


def flip_exception_type(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    patterns = [
        (r"catch\s*\(\s*([A-Za-z_$][\w$.]*)", "catch (Exception"),
        (r"except\s+([A-Za-z_$][\w$.]*)", "except Exception"),
        (r"rescue\s+([A-Za-z_:][\w:]*)", "rescue StandardError"),
    ]
    for pattern, replacement in patterns:
        match = re.search(pattern, code)
        if match:
            new = code[: match.start()] + replacement + code[match.end() :]
            return result(adapter, "flip_exception_type", "negative", code, new, "likely", {"negative_type": "wrong_exception_handling"})
    return None


def drop_keyword_argument(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    patterns = [r",\s*[A-Za-z_$][\w$]*\s*:\s*[^,)}\n]+", r"\{\s*[A-Za-z_$][\w$]*\s*:\s*[^,}}\n]+,\s*", r",\s*[A-Za-z_$][\w$]*\s*=\s*[^,)}\n]+"]
    for pattern in patterns:
        match = re.search(pattern, code)
        if match:
            new = code[: match.start()] + code[match.end() :]
            return result(adapter, "drop_keyword_argument", "negative", code, new, "likely", {"negative_type": "wrong_api_call"})
    return None


def copy_to_alias_mutation(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    replacements = [
        (r"\.copy\(\)", ""),
        (r"\.clone\(\)", ""),
        (r"\.dup\b", ""),
        (r"clone\s+", ""),
        (r"\[\.\.\.([^\]]+)\]", r"\1"),
        (r"new\s+ArrayList<>\(([^)]+)\)", r"\1"),
    ]
    for pattern, replacement in replacements:
        match = re.search(pattern, code)
        if match:
            new = re.sub(pattern, replacement, code, count=1)
            return result(adapter, "copy_to_alias_mutation", "negative", code, new, "likely", {"negative_type": "mutate_copy_vs_original"})
    return None


def drop_resource_context(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    patterns = [
        r"defer\s+[^\n;]+(?:\n|;)",
        r"using\s+[^\n;]+(?:\n|;)",
        r"finally\s*\{[^{}]*(?:close|fclose|dispose)[^{}]*\}",
        r"ensure\s+[^\n]*(?:close|fclose|dispose)[\s\S]*?end",
        r"\.close\(\);?",
        r"fclose\([^)]+\);?",
    ]
    for pattern in patterns:
        match = re.search(pattern, code, re.S)
        if match:
            new = code[: match.start()] + code[match.end() :]
            return result(adapter, "drop_resource_context", "negative", code, new, "likely", {"negative_type": "missing_resource_handling"})
    match = re.search(r"try\s*\([^)]*\)\s*\{", code)
    if match:
        new = code[: match.start()] + "try {" + code[match.end() :]
        return result(adapter, "drop_resource_context", "negative", code, new, "likely", {"negative_type": "missing_resource_handling"})
    return None


def replace_operator(adapter: GenericTreeSitterAdapter, code: str, name: str, mapping: dict[str, str], negative_type: str) -> TransformResult | None:
    for old in sorted(mapping, key=len, reverse=True):
        match = re.search(rf"(?<![=!<>]){re.escape(old)}(?![=])", code)
        if match:
            new = code[: match.start()] + mapping[old] + code[match.end() :]
            return result(adapter, name, "negative", code, new, "likely", {"negative_type": negative_type})
    return None


def simple_assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip().rstrip(";")
    if any(op in stripped for op in ["==", "!=", "<=", ">=", "+=", "-=", "*=", "/="]):
        return None
    match = re.match(r"(?:const|let|var|int|long|double|float|String|boolean|bool|def)?\s*([A-Za-z_$][\w$]*)\s*=\s*(.+)$", stripped)
    if not match:
        return None
    return match.group(1), match.group(2)


def identifiers(text: str, adapter: GenericTreeSitterAdapter) -> set[str]:
    return {m.group(0) for m in IDENT_RE.finditer(text) if not adapter._is_keyword(m.group(0))}


def strip_strings_and_comments(code: str, config: LanguageConfig) -> str:
    return STRING_RE.sub(lambda m: " " * (m.end() - m.start()), strip_comments_preserve_newlines(code, config))


def strip_comments_preserve_newlines(code: str, config: LanguageConfig) -> str:
    out = code
    if config.block_comment_start and config.block_comment_end:
        pattern = re.escape(config.block_comment_start) + r"[\s\S]*?" + re.escape(config.block_comment_end)
        out = re.sub(pattern, lambda m: "\n" * m.group(0).count("\n"), out)
    if config.line_comment:
        line_re = re.compile(re.escape(config.line_comment) + r".*")
        out = "\n".join(line_re.sub("", line) for line in out.splitlines()) + ("\n" if out.endswith("\n") else "")
    return out


def changed_span(old: str, new: str) -> dict[str, Any]:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    max_len = max(len(old_lines), len(new_lines))
    for index in range(max_len):
        left = old_lines[index] if index < len(old_lines) else ""
        right = new_lines[index] if index < len(new_lines) else ""
        if left != right:
            return {
                "node_type": "text_edit",
                "start_line": index + 1,
                "start_col": 0,
                "end_line": index + 1,
                "end_col": max(len(left), 1),
                "start_byte": len(("\n".join(old_lines[:index]) + ("\n" if index else "")).encode("utf-8")),
                "end_byte": len(("\n".join(old_lines[: index + 1])).encode("utf-8")),
                "source": left[:240],
            }
    return {"node_type": "text_edit", "start_line": 1, "start_col": 0, "end_line": 1, "end_col": 1, "start_byte": 0, "end_byte": 1, "source": old[:240]}


def balanced_text(code: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    for ch in STRING_RE.sub("", code):
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack.pop() != ch:
                return False
    return not stack


def line_col_from_offset(code: str, offset: int) -> tuple[int, int]:
    prefix = code[:offset]
    line = prefix.count("\n") + 1
    last = prefix.rfind("\n")
    col = offset if last < 0 else offset - last - 1
    return line, col


def walk_nodes(root: Any):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        try:
            count = int(node.named_child_count())
            children = [node.named_child(i) for i in range(count)]
        except Exception:
            children = []
        stack.extend(reversed([child for child in children if child is not None]))


def node_kind(node: Any) -> str:
    value = getattr(node, "kind", None)
    return str(value() if callable(value) else value)


def node_start_byte(node: Any) -> int:
    value = getattr(node, "start_byte", None)
    return int(value() if callable(value) else value)


def node_end_byte(node: Any) -> int:
    value = getattr(node, "end_byte", None)
    return int(value() if callable(value) else value)


def node_start_line(node: Any) -> int:
    point = node.start_position()
    row = getattr(point, "row", None)
    if callable(row):
        row = row()
    return int(row or 0) + 1


def node_end_line(node: Any) -> int:
    point = node.end_position()
    row = getattr(point, "row", None)
    if callable(row):
        row = row()
    return int(row or 0) + 1


def byte_slice(text: str, start: int, end: int) -> str:
    return text.encode("utf-8")[start:end].decode("utf-8", errors="ignore")


def function_node_kinds(language: str) -> set[str]:
    return {
        "python": {"function_definition"},
        "java": {"method_declaration", "constructor_declaration"},
        "javascript": {"function_declaration", "method_definition", "arrow_function", "function"},
        "go": {"function_declaration", "method_declaration"},
        "php": {"function_definition", "method_declaration"},
        "ruby": {"method", "singleton_method"},
    }.get(language, set())


def node_name(node: Any, segment: str) -> str:
    try:
        child = node.child_by_field_name("name")
        if child is not None:
            return byte_slice(segment, node_start_byte(child) - node_start_byte(node), node_end_byte(child) - node_start_byte(node))
    except Exception:
        pass
    match = re.search(r"\b(?:def|func|function)\s+([A-Za-z_$][\w$]*)", segment)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-Za-z_$][\w$]*)\s*\(", segment)
    return match.group(1) if match else ""


def find_matching_brace(code: str, open_index: int) -> int | None:
    depth = 0
    for index in range(open_index, len(code)):
        char = code[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


# Ruby-specific fallbacks kept here so Ruby still implements every canonical family.
def ruby_bool_return_simplify(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    if adapter.language != "ruby":
        return None
    match = re.search(r"if\s+([^\n]+)\n\s*(true|false)\s*\n\s*else\s*\n\s*(true|false)\s*\n\s*end", code)
    if not match:
        return None
    cond, left, right = match.groups()
    repl = cond.strip() if left == "true" and right == "false" else f"!({cond.strip()})" if left == "false" and right == "true" else ""
    if not repl:
        return None
    return result(adapter, "bool_return_simplify", "positive", code, code[: match.start()] + repl + code[match.end() :], "safe")


def ruby_if_return_merge(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    if adapter.language != "ruby":
        return None
    match = re.search(r"if\s+([^\n]+)\n\s*return\s+([^\n]+)\n\s*else\s*\n\s*return\s+([^\n]+)\n\s*end", code)
    if not match:
        return None
    cond, left, right = [item.strip() for item in match.groups()]
    repl = f"return {cond} ? {left} : {right}"
    return result(adapter, "if_return_merge", "positive", code, code[: match.start()] + repl + code[match.end() :], "likely")


def ruby_remove_unreachable_else(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    if adapter.language != "ruby":
        return None
    match = re.search(r"if\s+([^\n]+)\n(\s*return[^\n]+)\n\s*else\s*\n([\s\S]*?)\n\s*end", code)
    if not match:
        return None
    repl = f"if {match.group(1)}\n{match.group(2)}\nend\n{match.group(3).strip()}"
    return result(adapter, "remove_unreachable_else", "positive", code, code[: match.start()] + repl + code[match.end() :], "safe")


def ruby_negate_condition(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    if adapter.language != "ruby":
        return None
    match = re.search(r"\b(if|while|unless)\s+([^\n]+)", code)
    if not match:
        return None
    keyword = "if" if match.group(1) != "if" else "unless"
    repl = f"{keyword} {match.group(2).strip()}"
    return result(adapter, "negate_condition", "negative", code, code[: match.start()] + repl + code[match.end() :], "likely", {"negative_type": "condition_negation"})


def ruby_range_loop_to_while(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    if adapter.language != "ruby":
        return None
    match = re.search(r"for\s+(\w+)\s+in\s+([^\n]+)\.\.\.([^\n]+)\s*\n([\s\S]*?)\n\s*end", code)
    if not match:
        return None
    var, start, stop, body = [item.strip() for item in match.groups()]
    repl = f"{var} = {start}\nwhile {var} < {stop}\n{body}\n  {var} += 1\nend"
    return result(adapter, "range_loop_to_while_or_equivalent", "positive", code, code[: match.start()] + repl + code[match.end() :], "likely")


def ruby_demorgan_rewrite(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    if adapter.language != "ruby":
        return None
    match = re.search(r"!\s*\(([^\n()]+)\s+(and|or|&&|\|\|)\s+([^\n()]+)\)", code)
    if not match:
        return None
    left, op, right = [item.strip() for item in match.groups()]
    new_op = "||" if op in {"and", "&&"} else "&&"
    repl = f"!({left}) {new_op} !({right})"
    return result(adapter, "demorgan_rewrite", "positive", code, code[: match.start()] + repl + code[match.end() :], "safe")


def ruby_remove_guard_branch(adapter: GenericTreeSitterAdapter, code: str) -> TransformResult | None:
    if adapter.language != "ruby":
        return None
    match = re.search(r"\s*(?:return|raise)\s+[^\n]+\s+if\s+[^\n]+\n", code)
    if not match:
        return None
    return result(adapter, "remove_guard_branch", "negative", code, code[: match.start()] + code[match.end() :], "likely", {"negative_type": "missing_edge_case_branch"})
