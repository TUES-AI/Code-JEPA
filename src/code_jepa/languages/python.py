"""Python language adapter backed by the existing conservative AST transforms."""

from __future__ import annotations

import ast
import textwrap
from typing import Any

from code_jepa.data.ids import stable_hash
from code_jepa.data.python_ast import ParseResult, ast_spans, line_count, loc_bucket, parse_and_compile, rough_token_len
from code_jepa.languages.base import CANONICAL_TRANSFORM_INVENTORY, LanguageUnit, ParseStatus, role_transform_names
from code_jepa.transforms.python_ast import TransformResult
from code_jepa.transforms import python_ast as py_transforms

PY_POSITIVE_NAME_MAP = {
    "ast_normalize": "surface_normalize",
    "remove_docstrings": "remove_comments_or_docstrings",
    "rename_locals": "rename_locals",
    "swap_independent_assignments": "swap_independent_assignments",
    "bool_return_simplify": "bool_return_simplify",
    "if_return_to_conditional_expr": "if_return_merge",
    "remove_unreachable_else": "remove_unreachable_else",
    "safe_import_sort_same_block": "import_sort_same_block",
    "for_range_to_while": "range_loop_to_while_or_equivalent",
    "list_append_loop_to_comprehension": "append_loop_to_collection_literal",
    "accumulator_loop_to_sum": "accumulator_loop_to_builtin",
    "demorgan_boolean_rewrite": "demorgan_rewrite",
    "swap_independent_statement_block": "swap_independent_statement_block",
}

PY_NEGATIVE_NAME_MAP = {
    "flip_comparison": "flip_comparison",
    "flip_boolop": "flip_boolop",
    "swap_call_args": "swap_call_args",
    "wrong_variable": "wrong_variable",
    "flip_small_integer": "flip_small_integer",
    "flip_membership_or_identity": "flip_membership_or_identity",
    "negate_condition": "negate_condition",
    "flip_arithmetic_operator": "flip_arithmetic_operator",
    "flip_subscript_index": "flip_subscript_index",
    "flip_default_value": "flip_default_value",
    "flip_sort_reverse": "flip_sort_reverse",
    "remove_return_value": "remove_return_value",
    "drop_await": "remove_async_or_concurrency_wait",
    "shift_range_bound": "shift_loop_bound",
    "remove_guard_branch": "remove_guard_branch",
    "flip_exception_type": "flip_exception_type",
    "drop_keyword_argument": "drop_keyword_argument",
    "replace_copy_with_alias": "copy_to_alias_mutation",
    "drop_context_manager": "drop_resource_context",
}


class PythonAdapter:
    language = "python"

    def __init__(self) -> None:
        self.implemented_transforms = {
            stage: {role: set(names) for role, names in roles.items()}
            for stage, roles in CANONICAL_TRANSFORM_INVENTORY.items()
        }

    def parse(self, code: str) -> ParseStatus:
        parsed = parse_and_compile(code)
        return ParseStatus(parsed.tree, parsed.parse_ok, parsed.compile_ok, parsed.error)

    def extract_units(self, file_id: str, source: str, cfg: Any) -> list[LanguageUnit]:
        parsed = parse_and_compile(source)
        if parsed.tree is None:
            return []
        units: list[LanguageUnit] = []
        imports_context = self.imports_from_tree(parsed.tree, source)
        for node in getattr(parsed.tree, "body", []):
            if isinstance(node, ast.ClassDef):
                item = self.class_summary_unit(node, imports_context)
                units.append(item)
                if len(units) >= cfg.max_units_per_file:
                    return units
        extractor = _UnitExtractor(source, imports_context, cfg.max_units_per_file - len(units))
        extractor.visit(parsed.tree)
        units.extend(extractor.units)
        if not units:
            units.append(self._whole_source_unit(source))
        if len(units) < cfg.max_units_per_file:
            units.extend(self.file_window_units(source, imports_context, cfg, remaining=cfg.max_units_per_file - len(units)))
        return units[: cfg.max_units_per_file]

    def _whole_source_unit(self, source: str) -> LanguageUnit:
        return LanguageUnit(
            unit_family="function",
            unit_type="function",
            qualified_name="codesearchnet_row",
            code=source.rstrip() + "\n",
            start_line=1,
            end_line=line_count(source),
            identifiers=self.identifiers_from_code(source),
            calls=self.calls_from_code(source),
            ast_sequence=self.ast_sequence_from_code(source),
        )

    def class_summary_unit(self, node: ast.ClassDef, imports_context: str) -> LanguageUnit:
        signatures = [function_signature(child) for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
        bases = [safe_unparse(base) for base in node.bases]
        suffix = f"({', '.join(bases)})" if bases else ""
        code = f"class {node.name}{suffix}:\n    ..."
        if signatures:
            code += "\n" + "\n".join(f"    {sig}" for sig in signatures)
        code = code.rstrip() + "\n"
        return LanguageUnit(
            unit_family="class",
            unit_type="class_summary",
            qualified_name=node.name,
            code=code,
            start_line=int(getattr(node, "lineno", 0)),
            end_line=int(getattr(node, "end_lineno", getattr(node, "lineno", 0))),
            imports_context=imports_context,
            class_context=class_signature(node),
            sibling_signatures="\n".join(signatures),
            identifiers=identifiers_from_tree(node),
            calls=calls_from_tree(node),
            ast_sequence=ast_sequence_from_tree(node),
        )

    def file_window_units(self, source: str, imports_context: str, cfg: Any, *, remaining: int) -> list[LanguageUnit]:
        lines = source.splitlines()
        out = []
        if len(lines) < cfg.file_window_lines:
            return out
        for index, start in enumerate(range(0, len(lines), cfg.file_window_stride)):
            if len(out) >= min(remaining, cfg.max_file_windows):
                break
            end = min(len(lines), start + cfg.file_window_lines)
            window = "\n".join(lines[start:end]).rstrip() + "\n"
            if not window.strip():
                continue
            out.append(
                LanguageUnit(
                    unit_family="file_window",
                    unit_type="file_window",
                    qualified_name=f"file_window_{index:03d}",
                    code=window,
                    start_line=start + 1,
                    end_line=end,
                    imports_context=imports_context,
                    identifiers=self.identifiers_from_code(window),
                    calls=self.calls_from_code(window),
                    ast_sequence=self.ast_sequence_from_code(window),
                )
            )
        return out

    def spans(self, unit_id: str, code: str, max_spans: int) -> list[dict[str, Any]]:
        return ast_spans(unit_id, code)[:max_spans]

    def positive_views_for_stage(
        self,
        code: str,
        stage: str,
        *,
        max_views: int = 16,
        only_transform: str = "",
    ) -> list[TransformResult]:
        return self._mapped_views(code, stage, "positive", max_views=max_views, only_transform=only_transform)

    def negative_views_for_stage(
        self,
        code: str,
        stage: str,
        *,
        max_views: int = 8,
        only_transform: str = "",
    ) -> list[TransformResult]:
        return self._mapped_views(code, stage, "negative", max_views=max_views, only_transform=only_transform)

    def _mapped_views(
        self,
        code: str,
        stage: str,
        role: str,
        *,
        max_views: int,
        only_transform: str,
    ) -> list[TransformResult]:
        if role == "positive":
            raw = py_transforms.positive_views_for_stage(code, stage, max_views=64)
            mapping = PY_POSITIVE_NAME_MAP
        else:
            raw = py_transforms.hard_negative_views_for_stage(code, stage, max_views=64)
            mapping = PY_NEGATIVE_NAME_MAP
        allowed = set(role_transform_names(stage, role, only_transform=only_transform))
        out = []
        for item in raw:
            canonical = mapping.get(item.name, item.name)
            if canonical not in allowed:
                continue
            out.append(rename_transform(item, canonical, self.language))
            if len(out) >= max_views:
                break
        return out

    def imports_context(self, code: str) -> str:
        parsed = parse_and_compile(code)
        if parsed.tree is None:
            return ""
        return self.imports_from_tree(parsed.tree, code)

    def imports_from_tree(self, tree: ast.AST, source: str) -> str:
        lines = []
        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                segment = ast.get_source_segment(source, node)
                if segment:
                    lines.append(segment)
            elif not isinstance(node, ast.Expr):
                break
        return "\n".join(lines[:128])

    def identifiers_from_code(self, code: str) -> list[str]:
        parsed = parse_and_compile(code)
        return identifiers_from_tree(parsed.tree) if parsed.tree is not None else []

    def calls_from_code(self, code: str) -> list[str]:
        parsed = parse_and_compile(code)
        return calls_from_tree(parsed.tree) if parsed.tree is not None else []

    def ast_sequence_from_code(self, code: str) -> list[str]:
        parsed = parse_and_compile(code)
        return ast_sequence_from_tree(parsed.tree) if parsed.tree is not None else []


def rename_transform(item: TransformResult, canonical_name: str, language: str) -> TransformResult:
    metadata = dict(item.metadata)
    metadata["source_transform_name"] = item.name
    metadata["stage_family_name"] = canonical_name
    metadata["language"] = language
    return TransformResult(
        name=canonical_name,
        role=item.role,
        code=item.code,
        confidence=item.confidence,
        changed_spans=item.changed_spans,
        metadata=metadata,
    )


class _UnitExtractor(ast.NodeVisitor):
    def __init__(self, source: str, imports_context: str, max_units: int) -> None:
        self.source = source
        self.imports_context = imports_context
        self.max_units = max_units
        self.units: list[LanguageUnit] = []
        self.class_stack: list[ast.ClassDef] = []
        self.function_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node)
        for child in node.body:
            self.visit(child)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_function(node)
        self.function_stack.append(node)
        for child in node.body:
            self.visit(child)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_function(node)
        self.function_stack.append(node)
        for child in node.body:
            self.visit(child)
        self.function_stack.pop()

    def _add_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if len(self.units) >= self.max_units:
            return
        segment = ast.get_source_segment(self.source, node)
        if not segment:
            return
        class_names = [item.name for item in self.class_stack]
        parent_funcs = [item.name for item in self.function_stack]
        qualified_name = ".".join([*class_names, *parent_funcs, node.name])
        if self.function_stack:
            unit_family = "nested_function"
        elif self.class_stack:
            unit_family = "method"
        else:
            unit_family = "function"
        code = textwrap.dedent(segment).rstrip() + "\n"
        self.units.append(
            LanguageUnit(
                unit_family=unit_family,
                unit_type=unit_family,
                qualified_name=qualified_name or node.name,
                code=code,
                start_line=int(getattr(node, "lineno", 0)),
                end_line=int(getattr(node, "end_lineno", 0)),
                imports_context=self.imports_context,
                class_context=class_signature(self.class_stack[-1]) if self.class_stack else "",
                sibling_signatures=sibling_signatures(self.class_stack[-1] if self.class_stack else None, node),
                identifiers=identifiers_from_tree(node),
                calls=calls_from_tree(node),
                ast_sequence=ast_sequence_from_tree(node),
            )
        )


def class_signature(node: ast.ClassDef) -> str:
    bases = [safe_unparse(base) for base in node.bases]
    suffix = f"({', '.join(bases)})" if bases else ""
    return f"class {node.name}{suffix}: ..."


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = "..."
    return f"{prefix} {node.name}({args}): ..."


def sibling_signatures(class_node: ast.ClassDef | None, current: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    if class_node is None:
        return ""
    signatures = []
    for child in class_node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name != current.name:
            signatures.append(function_signature(child))
    return "\n".join(signatures[:64])


def identifiers_from_tree(tree: ast.AST) -> list[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return sorted(names)[:512]


def calls_from_tree(tree: ast.AST) -> list[str]:
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = call_name(node.func)
            if name and name not in calls:
                calls.append(name)
    return calls[:512]


def ast_sequence_from_tree(tree: ast.AST) -> list[str]:
    return [type(node).__name__ for node in ast.walk(tree)][:1024]


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "..."
