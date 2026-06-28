"""Conservative Python AST transforms for Code-JEPA data prep."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from typing import Any

from code_jepa.data.python_ast import node_span, parse_and_compile, unparse


@dataclass(frozen=True)
class TransformResult:
    name: str
    role: str
    code: str
    confidence: str
    changed_spans: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def compile_valid(code: str) -> bool:
    return parse_and_compile(code).compile_ok


def positive_views(code: str, *, max_views: int = 3) -> list[TransformResult]:
    """Compatibility helper: conservative v0 positives only."""

    return positive_views_for_stage(code, "v0", max_views=max_views)


def positive_views_for_stage(code: str, stage: str, *, max_views: int = 16) -> list[TransformResult]:
    """Return stage-local positive views for one transform stage."""

    if stage not in TRANSFORM_STAGES:
        raise ValueError(f"unknown transform stage {stage!r}; expected one of {TRANSFORM_STAGES}")
    parsed = parse_and_compile(code)
    if parsed.tree is None:
        return []

    if stage == "v0":
        candidates = _v0_positive_candidates(parsed.tree, code)
    elif stage == "v1":
        candidates = _v1_positive_candidates(parsed.tree, code)
    else:
        candidates = _v2_positive_candidates(parsed.tree, code)
    return _dedupe_valid(code, candidates, max_views=max_views)


def _v0_positive_candidates(tree: ast.AST, code: str) -> list[TransformResult | None]:
    del code
    return [
        _safe_transform(_ast_normalized, tree),
        _safe_transform(_remove_docstrings, tree),
        _safe_transform(_rename_locals, tree),
    ]


def _v1_positive_candidates(tree: ast.AST, code: str) -> list[TransformResult | None]:
    return [
        _safe_transform(_swap_independent_assignments, tree, code),
        _safe_transform(_bool_return_simplify, tree, code),
        _safe_transform(_if_return_to_conditional_expr, tree, code),
        _safe_transform(_remove_unreachable_else, tree, code),
        _safe_transform(_safe_import_sort_same_block, tree, code),
    ]


def _v2_positive_candidates(tree: ast.AST, code: str) -> list[TransformResult | None]:
    return [
        _safe_transform(_for_range_to_while, tree, code),
        _safe_transform(_list_append_loop_to_comprehension, tree, code),
        _safe_transform(_accumulator_loop_to_sum, tree, code),
        _safe_transform(_demorgan_boolean_rewrite, tree, code),
        _safe_transform(_swap_independent_statement_block, tree, code),
    ]


TRANSFORM_STAGES = ("v0", "v1", "v2")


def hard_negative_views(code: str, *, max_views: int = 6) -> list[TransformResult]:
    """Compatibility helper: v0 plus v1 harder negatives."""

    parsed = parse_and_compile(code)
    if parsed.tree is None:
        return []
    return _dedupe_valid(
        code,
        [*_v0_negative_candidates(parsed.tree, code), *_v1_negative_candidates(parsed.tree, code)],
        max_views=max_views,
    )


def extra_hard_negative_views(code: str, *, max_views: int = 8) -> list[TransformResult]:
    parsed = parse_and_compile(code)
    if parsed.tree is None:
        return []
    return _dedupe_valid(code, _v1_negative_candidates(parsed.tree, code), max_views=max_views)


def hard_negative_views_for_stage(code: str, stage: str, *, max_views: int = 8) -> list[TransformResult]:
    """Return reproducible negative views for one transform stage.

    `v0`, `v1`, and `v2` are transformation-family names, not data-pipeline versions.
    The data-prep pipeline can emit each stage into a separate segment and combine them later.
    """

    if stage not in TRANSFORM_STAGES:
        raise ValueError(f"unknown transform stage {stage!r}; expected one of {TRANSFORM_STAGES}")
    parsed = parse_and_compile(code)
    if parsed.tree is None:
        return []
    if stage == "v0":
        candidates = _v0_negative_candidates(parsed.tree, code)
    elif stage == "v1":
        candidates = _v1_negative_candidates(parsed.tree, code)
    else:
        candidates = _v2_negative_candidates(parsed.tree, code)
    return _dedupe_valid(code, candidates, max_views=max_views)


def _v0_negative_candidates(tree: ast.AST, code: str) -> list[TransformResult | None]:
    return [
        _safe_transform(_flip_comparison, tree, code),
        _safe_transform(_flip_boolop, tree, code),
        _safe_transform(_swap_call_args, tree, code),
        _safe_transform(_wrong_variable, tree, code),
        _safe_transform(_flip_small_integer, tree, code),
    ]


def _v1_negative_candidates(tree: ast.AST, code: str) -> list[TransformResult | None]:
    return [
        _safe_transform(_flip_membership_or_identity, tree, code),
        _safe_transform(_negate_first_condition, tree, code),
        _safe_transform(_flip_arithmetic_operator, tree, code),
        _safe_transform(_flip_subscript_index, tree, code),
        _safe_transform(_flip_default_value, tree, code),
        _safe_transform(_flip_sort_reverse, tree, code),
        _safe_transform(_remove_return_value, tree, code),
        _safe_transform(_drop_await, tree, code),
    ]


def _harder_negative_candidates(tree: ast.AST, code: str) -> list[TransformResult | None]:
    return _v1_negative_candidates(tree, code)


def _v2_negative_candidates(tree: ast.AST, code: str) -> list[TransformResult | None]:
    return [
        _safe_transform(_shift_range_bound, tree, code),
        _safe_transform(_remove_guard_branch, tree, code),
        _safe_transform(_flip_exception_type, tree, code),
        _safe_transform(_drop_keyword_argument, tree, code),
        _safe_transform(_replace_copy_call_with_alias, tree, code),
        _safe_transform(_drop_context_manager, tree, code),
    ]


def _safe_transform(fn, *args) -> TransformResult | None:
    try:
        return fn(*args)
    except (RecursionError, MemoryError):
        return None


def _dedupe_valid(
    anchor_code: str, candidates: list[TransformResult | None], *, max_views: int
) -> list[TransformResult]:
    out: list[TransformResult] = []
    seen = {anchor_code.strip()}
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = candidate.code.strip()
        if normalized in seen:
            continue
        if not compile_valid(candidate.code):
            continue
        seen.add(normalized)
        out.append(candidate)
        if len(out) >= max_views:
            break
    return out


def _ast_normalized(tree: ast.AST) -> TransformResult:
    new_tree = copy.deepcopy(tree)
    return TransformResult(
        name="ast_normalize",
        role="positive",
        code=unparse(new_tree),
        confidence="safe",
        metadata={"positive_type": "format_comment_normalization"},
    )


class _DocstringRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = _without_leading_docstring(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = _without_leading_docstring(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        node.body = _without_leading_docstring(node.body)
        return node


def _without_leading_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if not body:
        return body
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return body[1:] or [ast.Pass()]
    return body


def _remove_docstrings(tree: ast.AST) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    before = unparse(new_tree)
    _DocstringRemover().visit(new_tree)
    after = unparse(new_tree)
    if after.strip() == before.strip():
        return None
    return TransformResult(
        name="remove_docstrings",
        role="positive",
        code=after,
        confidence="likely",
        metadata={"positive_type": "docstring_removed"},
    )


class _LocalNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.blocked: set[str] = set()

    def visit_Global(self, node: ast.Global) -> None:
        self.blocked.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.blocked.update(node.names)

    def visit_arg(self, node: ast.arg) -> None:
        if node.arg not in {"self", "cls"}:
            self.bound.add(node.arg)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.bound.add(node.id)


class _LocalRenamer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node


def _rename_locals(tree: ast.AST) -> TransformResult | None:
    collector = _LocalNameCollector()
    collector.visit(tree)
    names = sorted(
        name
        for name in collector.bound - collector.blocked
        if name.isidentifier() and not (name.startswith("__") and name.endswith("__"))
    )[:8]
    if not names:
        return None
    mapping = {name: f"cjepa_{i}" for i, name in enumerate(names)}
    new_tree = copy.deepcopy(tree)
    _LocalRenamer(mapping).visit(new_tree)
    return TransformResult(
        name="rename_locals",
        role="positive",
        code=unparse(new_tree),
        confidence="likely",
        metadata={"mapping": mapping, "positive_type": "local_rename"},
    )


_CMP_FLIPS: dict[type[ast.cmpop], ast.cmpop] = {
    ast.Lt: ast.LtE(),
    ast.LtE: ast.Lt(),
    ast.Gt: ast.GtE(),
    ast.GtE: ast.Gt(),
    ast.Eq: ast.NotEq(),
    ast.NotEq: ast.Eq(),
}


class _FirstCompareFlip(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if self.changed:
            return node
        for index, op in enumerate(node.ops):
            replacement = _CMP_FLIPS.get(type(op))
            if replacement is None:
                continue
            old_op = type(op).__name__
            node.ops[index] = replacement
            span = node_span(node, self.source_code) or {}
            span.update(
                {
                    "kind": "comparison_operator",
                    "original": old_op,
                    "replacement": type(replacement).__name__,
                }
            )
            self.changed.append(span)
            break
        return node


def _flip_comparison(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstCompareFlip(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="flip_comparison",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "comparison_operator"},
    )


class _FirstBoolOpFlip(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        if isinstance(node.op, ast.And):
            old, node.op = "And", ast.Or()
        elif isinstance(node.op, ast.Or):
            old, node.op = "Or", ast.And()
        else:
            return node
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "boolean_operator", "original": old, "replacement": type(node.op).__name__})
        self.changed.append(span)
        return node


def _flip_boolop(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstBoolOpFlip(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="flip_boolop",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "boolean_operator"},
    )


class _FirstCallArgSwap(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if self.changed or len(node.args) < 2:
            return node
        node.args[0], node.args[1] = node.args[1], node.args[0]
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "call_arg_swap", "swapped_arg_indices": [0, 1]})
        self.changed.append(span)
        return node


def _swap_call_args(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstCallArgSwap(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="swap_call_args",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "swapped_args"},
    )


class _LoadNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: list[str] = []

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id not in self.names:
            self.names.append(node.id)


class _FirstWrongVariable(ast.NodeTransformer):
    def __init__(self, source_code: str, mapping: dict[str, str]) -> None:
        self.source_code = source_code
        self.mapping = mapping
        self.changed: list[dict[str, Any]] = []

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self.changed or not isinstance(node.ctx, ast.Load) or node.id not in self.mapping:
            return node
        old = node.id
        node.id = self.mapping[old]
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "wrong_variable", "original": old, "replacement": node.id})
        self.changed.append(span)
        return node


def _wrong_variable(tree: ast.AST, code: str) -> TransformResult | None:
    collector = _LoadNameCollector()
    collector.visit(tree)
    names = [name for name in collector.names if name not in {"self", "cls", "True", "False", "None"}]
    if len(names) < 2:
        return None
    mapping = {names[0]: names[1]}
    new_tree = copy.deepcopy(tree)
    mutator = _FirstWrongVariable(code, mapping)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="wrong_variable",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "wrong_variable", "mapping": mapping},
    )


class _FirstSmallIntegerFlip(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.changed or not isinstance(node.value, int) or isinstance(node.value, bool):
            return node
        if node.value not in {0, 1, -1}:
            return node
        old = node.value
        node.value = 0 if old == 1 else 1
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "small_integer", "original": old, "replacement": node.value})
        self.changed.append(span)
        return node


def _flip_small_integer(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstSmallIntegerFlip(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="flip_small_integer",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "small_integer"},
    )


_MEMBERSHIP_FLIPS: dict[type[ast.cmpop], ast.cmpop] = {
    ast.In: ast.NotIn(),
    ast.NotIn: ast.In(),
    ast.Is: ast.IsNot(),
    ast.IsNot: ast.Is(),
}


class _FirstMembershipOrIdentityFlip(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if self.changed:
            return node
        for index, op in enumerate(node.ops):
            replacement = _MEMBERSHIP_FLIPS.get(type(op))
            if replacement is None:
                continue
            old_op = type(op).__name__
            node.ops[index] = replacement
            span = node_span(node, self.source_code) or {}
            span.update({"kind": "membership_or_identity", "original": old_op, "replacement": type(replacement).__name__})
            self.changed.append(span)
            break
        return node


def _flip_membership_or_identity(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstMembershipOrIdentityFlip(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="flip_membership_or_identity",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "membership_or_identity"},
    )


class _FirstConditionNegator(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        old_kind = type(node.test).__name__
        node.test = _negated_expr(node.test)
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "condition_negation", "original": old_kind})
        self.changed.append(span)
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        old_kind = type(node.test).__name__
        node.test = _negated_expr(node.test)
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "condition_negation", "original": old_kind})
        self.changed.append(span)
        return node


def _negated_expr(expr: ast.expr) -> ast.expr:
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return expr.operand
    return ast.UnaryOp(op=ast.Not(), operand=expr)


def _negate_first_condition(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstConditionNegator(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="negate_condition",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "condition_negation"},
    )


_BINOP_FLIPS: dict[type[ast.operator], ast.operator] = {
    ast.Add: ast.Sub(),
    ast.Sub: ast.Add(),
    ast.Mult: ast.FloorDiv(),
    ast.FloorDiv: ast.Mult(),
    ast.Mod: ast.Mult(),
}


class _FirstArithmeticFlip(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        replacement = _BINOP_FLIPS.get(type(node.op))
        if replacement is None:
            return node
        old_op = type(node.op).__name__
        node.op = replacement
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "arithmetic_operator", "original": old_op, "replacement": type(replacement).__name__})
        self.changed.append(span)
        return node


def _flip_arithmetic_operator(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstArithmeticFlip(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="flip_arithmetic_operator",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "arithmetic_operator"},
    )


class _FirstSubscriptIndexFlip(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        target = node.slice
        if not isinstance(target, ast.Constant) or not isinstance(target.value, int) or isinstance(target.value, bool):
            return node
        if target.value not in {0, 1, -1}:
            return node
        old = target.value
        target.value = 0 if old == 1 else 1
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "subscript_index", "original": old, "replacement": target.value})
        self.changed.append(span)
        return node


def _flip_subscript_index(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstSubscriptIndexFlip(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="flip_subscript_index",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "subscript_index"},
    )


def _flipped_constant(value: Any) -> Any | None:
    if isinstance(value, bool):
        return not value
    if value is None:
        return 0
    if isinstance(value, int):
        if value in {0, 1, -1}:
            return 0 if value == 1 else 1
        return value + 1
    if isinstance(value, str):
        return value + "_wrong"
    return None


class _FirstDefaultValueFlip(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        self._flip_default(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        self._flip_default(node)
        return node

    def _flip_default(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.changed:
            return
        for default in list(node.args.defaults) + list(node.args.kw_defaults):
            if not isinstance(default, ast.Constant):
                continue
            replacement = _flipped_constant(default.value)
            if replacement is None or replacement == default.value:
                continue
            old = default.value
            default.value = replacement
            span = node_span(node, self.source_code) or {}
            span.update({"kind": "default_value", "original": old, "replacement": replacement})
            self.changed.append(span)
            return


def _flip_default_value(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstDefaultValueFlip(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="flip_default_value",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "default_value"},
    )


class _FirstSortReverseFlip(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name not in {"sorted", "sort"}:
            return node
        for keyword in node.keywords:
            if keyword.arg == "reverse" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, bool):
                old = keyword.value.value
                keyword.value.value = not old
                span = node_span(node, self.source_code) or {}
                span.update({"kind": "sort_reverse", "original": old, "replacement": not old})
                self.changed.append(span)
                return node
        node.keywords.append(ast.keyword(arg="reverse", value=ast.Constant(value=True)))
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "sort_reverse", "original": None, "replacement": True})
        self.changed.append(span)
        return node


def _flip_sort_reverse(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstSortReverseFlip(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="flip_sort_reverse",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "sort_reverse"},
    )


class _FirstReturnValueRemover(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_Return(self, node: ast.Return) -> ast.AST:
        self.generic_visit(node)
        if self.changed or node.value is None:
            return node
        old_kind = type(node.value).__name__
        node.value = ast.Constant(value=None)
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "return_value", "original": old_kind, "replacement": "None"})
        self.changed.append(span)
        return node


def _remove_return_value(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstReturnValueRemover(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="remove_return_value",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "return_value_removed"},
    )


class _FirstAwaitDropper(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_Await(self, node: ast.Await) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "await_removed"})
        self.changed.append(span)
        return node.value


def _drop_await(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstAwaitDropper(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="drop_await",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "await_removed"},
    )


class _FirstRangeBoundShift(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if self.changed or not isinstance(node.func, ast.Name) or node.func.id != "range":
            return node
        if not node.args:
            return node
        target_index = 0 if len(node.args) == 1 else 1
        old_node = node.args[target_index]
        node.args[target_index] = _plus_one(old_node)
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "loop_bound", "argument_index": target_index, "replacement": "+1"})
        self.changed.append(span)
        return node


def _plus_one(expr: ast.expr) -> ast.expr:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
        return ast.Constant(value=expr.value + 1)
    return ast.BinOp(left=expr, op=ast.Add(), right=ast.Constant(value=1))


def _shift_range_bound(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstRangeBoundShift(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="shift_range_bound",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "loop_bound_off_by_one"},
    )


class _FirstGuardBranchRemover(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        if self.changed or not node.body:
            return node
        if not any(isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)) for stmt in node.body):
            return node
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "guard_branch_removed", "original_body_len": len(node.body)})
        node.body = [ast.Pass()]
        self.changed.append(span)
        return node


def _remove_guard_branch(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstGuardBranchRemover(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="remove_guard_branch",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "missing_edge_case_branch"},
    )


_EXCEPTION_FLIPS = {
    "Exception": "ValueError",
    "ValueError": "TypeError",
    "TypeError": "ValueError",
    "KeyError": "IndexError",
    "IndexError": "KeyError",
    "FileNotFoundError": "PermissionError",
    "PermissionError": "FileNotFoundError",
}


class _FirstExceptionTypeFlip(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        self.generic_visit(node)
        if self.changed or node.type is None:
            return node
        old_name = _exception_name(node.type)
        if old_name is None:
            return node
        replacement = _EXCEPTION_FLIPS.get(old_name, "TypeError")
        node.type = ast.Name(id=replacement, ctx=ast.Load())
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "exception_type", "original": old_name, "replacement": replacement})
        self.changed.append(span)
        return node


def _exception_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _flip_exception_type(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstExceptionTypeFlip(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="flip_exception_type",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "wrong_exception_handling"},
    )


class _FirstKeywordDropper(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        for index, keyword in enumerate(node.keywords):
            if keyword.arg is None:
                continue
            removed = keyword.arg
            del node.keywords[index]
            span = node_span(node, self.source_code) or {}
            span.update({"kind": "keyword_argument_dropped", "removed": removed})
            self.changed.append(span)
            break
        return node


def _drop_keyword_argument(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstKeywordDropper(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="drop_keyword_argument",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "wrong_api_call"},
    )


class _FirstCopyAliasReplacer(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if self.changed or not isinstance(node.func, ast.Attribute) or node.func.attr not in {"copy", "deepcopy"}:
            return node
        replacement: ast.expr | None = None
        if node.func.attr == "copy" and not node.args and not node.keywords:
            replacement = node.func.value
        elif node.args and _exception_name(node.func.value) in {"copy", "deepcopy"}:
            replacement = node.args[0]
        if replacement is None:
            return node
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "copy_to_alias", "method": node.func.attr})
        self.changed.append(span)
        return replacement


def _replace_copy_call_with_alias(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstCopyAliasReplacer(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="replace_copy_with_alias",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "mutate_copy_vs_original"},
    )


class _FirstContextManagerDropper(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_With(self, node: ast.With) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        if self.changed:
            return node
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "context_manager_dropped", "items": len(node.items)})
        self.changed.append(span)
        return node.body or [ast.Pass()]

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        if self.changed:
            return node
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "context_manager_dropped", "items": len(node.items), "async": True})
        self.changed.append(span)
        return node.body or [ast.Pass()]


def _drop_context_manager(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstContextManagerDropper(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="drop_context_manager",
        role="negative",
        code=unparse(new_tree),
        confidence="behavior_impacting",
        changed_spans=mutator.changed,
        metadata={"negative_type": "missing_resource_handling"},
    )


class _SingleBodyRewrite(ast.NodeTransformer):
    source_code: str
    changed: list[dict[str, Any]]

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        node.body = self._rewrite_body(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._rewrite_body(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._rewrite_body(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._rewrite_body(node.body)
        return node

    def _rewrite_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        return body


class _IndependentAssignmentSwapper(_SingleBodyRewrite):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed = []

    def _rewrite_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if self.changed:
            return body
        for index in range(len(body) - 1):
            first, second = body[index], body[index + 1]
            if not _simple_independent_assignments(first, second):
                continue
            new_body = list(body)
            new_body[index], new_body[index + 1] = second, first
            span = node_span(first, self.source_code) or {}
            span.update({"kind": "independent_assignment_reorder"})
            self.changed.append(span)
            return new_body
        return body


def _swap_independent_assignments(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _IndependentAssignmentSwapper(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="swap_independent_assignments",
        role="positive",
        code=unparse(new_tree),
        confidence="likely",
        changed_spans=mutator.changed,
        metadata={"positive_type": "independent_statement_reorder", "stage": "v1"},
    )


class _BoolReturnSimplifier(_SingleBodyRewrite):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed = []

    def _rewrite_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if self.changed:
            return body
        for index, stmt in enumerate(body):
            if not isinstance(stmt, ast.If):
                continue
            result = _bool_return_replacement(stmt, body[index + 1] if index + 1 < len(body) else None)
            if result is None:
                continue
            replacement, consume_next = result
            new_body = list(body)
            new_body[index] = ast.Return(value=replacement)
            if consume_next:
                del new_body[index + 1]
            span = node_span(stmt, self.source_code) or {}
            span.update({"kind": "bool_return_simplify"})
            self.changed.append(span)
            return new_body
        return body


def _bool_return_simplify(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _BoolReturnSimplifier(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="bool_return_simplify",
        role="positive",
        code=unparse(new_tree),
        confidence="safe",
        changed_spans=mutator.changed,
        metadata={"positive_type": "control_flow_simplification", "stage": "v1"},
    )


class _IfReturnConditionalRewriter(_SingleBodyRewrite):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed = []

    def _rewrite_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if self.changed:
            return body
        for index, stmt in enumerate(body):
            if not isinstance(stmt, ast.If) or len(stmt.body) != 1:
                continue
            if not isinstance(stmt.body[0], ast.Return) or stmt.body[0].value is None:
                continue
            else_value = None
            consume_next = False
            if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.Return):
                else_value = stmt.orelse[0].value
            elif not stmt.orelse and index + 1 < len(body) and isinstance(body[index + 1], ast.Return):
                else_value = body[index + 1].value
                consume_next = True
            if else_value is None:
                continue
            new_body = list(body)
            new_body[index] = ast.Return(value=ast.IfExp(test=stmt.test, body=stmt.body[0].value, orelse=else_value))
            if consume_next:
                del new_body[index + 1]
            span = node_span(stmt, self.source_code) or {}
            span.update({"kind": "if_return_to_conditional_expr"})
            self.changed.append(span)
            return new_body
        return body


def _if_return_to_conditional_expr(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _IfReturnConditionalRewriter(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="if_return_to_conditional_expr",
        role="positive",
        code=unparse(new_tree),
        confidence="safe",
        changed_spans=mutator.changed,
        metadata={"positive_type": "control_flow_simplification", "stage": "v1"},
    )


class _UnreachableElseRemover(_SingleBodyRewrite):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed = []

    def _rewrite_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if self.changed:
            return body
        for index, stmt in enumerate(body):
            if not isinstance(stmt, ast.If) or not stmt.orelse or not stmt.body:
                continue
            if not _terminal_stmt(stmt.body[-1]):
                continue
            new_if = copy.deepcopy(stmt)
            moved = new_if.orelse
            new_if.orelse = []
            new_body = [*body[:index], new_if, *moved, *body[index + 1 :]]
            span = node_span(stmt, self.source_code) or {}
            span.update({"kind": "unreachable_else_removed"})
            self.changed.append(span)
            return new_body
        return body


def _remove_unreachable_else(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _UnreachableElseRemover(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="remove_unreachable_else",
        role="positive",
        code=unparse(new_tree),
        confidence="safe",
        changed_spans=mutator.changed,
        metadata={"positive_type": "control_flow_simplification", "stage": "v1"},
    )


class _ModuleImportSorter(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        end = 0
        while end < len(node.body) and isinstance(node.body[end], (ast.Import, ast.ImportFrom)):
            end += 1
        if end < 2:
            return node
        block = node.body[:end]
        if any(isinstance(item, ast.ImportFrom) and item.module == "__future__" for item in block):
            return node
        sorted_block = sorted(block, key=lambda item: ast.unparse(item))
        if [ast.unparse(item) for item in block] == [ast.unparse(item) for item in sorted_block]:
            return node
        node.body = [*sorted_block, *node.body[end:]]
        span = node_span(block[0], self.source_code) or {}
        span.update({"kind": "safe_import_sort_same_block", "imports": end})
        self.changed.append(span)
        return node


def _safe_import_sort_same_block(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _ModuleImportSorter(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="safe_import_sort_same_block",
        role="positive",
        code=unparse(new_tree),
        confidence="likely",
        changed_spans=mutator.changed,
        metadata={"positive_type": "import_block_sort", "stage": "v1"},
    )


class _ForRangeToWhile(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_For(self, node: ast.For) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        if self.changed or node.orelse or not isinstance(node.target, ast.Name):
            return node
        if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name):
            return node
        if node.iter.func.id != "range" or len(node.iter.args) not in {1, 2} or node.iter.keywords:
            return node
        if _contains_node_type(node.body, ast.Continue) or _body_assigns_name(node.body, node.target.id):
            return node
        if len(node.iter.args) == 1:
            start = ast.Constant(value=0)
            stop = node.iter.args[0]
        else:
            start, stop = node.iter.args
        if not _stable_loop_bound(stop) or not _stable_loop_bound(start):
            return node
        if node.target.id in _load_names(stop) or node.target.id in _load_names(start):
            return node
        if isinstance(stop, ast.Name) and _body_assigns_name(node.body, stop.id):
            return node
        init = ast.Assign(targets=[ast.Name(id=node.target.id, ctx=ast.Store())], value=start)
        test = ast.Compare(left=ast.Name(id=node.target.id, ctx=ast.Load()), ops=[ast.Lt()], comparators=[stop])
        increment = ast.AugAssign(target=ast.Name(id=node.target.id, ctx=ast.Store()), op=ast.Add(), value=ast.Constant(value=1))
        while_node = ast.While(test=test, body=[*node.body, increment], orelse=[])
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "for_range_to_while"})
        self.changed.append(span)
        return [init, while_node]


def _for_range_to_while(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _ForRangeToWhile(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="for_range_to_while",
        role="positive",
        code=unparse(new_tree),
        confidence="likely",
        changed_spans=mutator.changed,
        metadata={"positive_type": "loop_refactor", "stage": "v2"},
    )


class _ListAppendLoopToComprehension(_SingleBodyRewrite):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed = []

    def _rewrite_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if self.changed:
            return body
        for index in range(len(body) - 2):
            assign, loop, ret = body[index], body[index + 1], body[index + 2]
            result = _list_append_comp_replacement(assign, loop, ret)
            if result is None:
                continue
            out_name, comp = result
            new_body = [*body[:index], ast.Return(value=comp), *body[index + 3 :]]
            span = node_span(loop, self.source_code) or {}
            span.update({"kind": "list_append_loop_to_comprehension", "accumulator": out_name})
            self.changed.append(span)
            return new_body
        return body


def _list_append_loop_to_comprehension(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _ListAppendLoopToComprehension(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="list_append_loop_to_comprehension",
        role="positive",
        code=unparse(new_tree),
        confidence="likely",
        changed_spans=mutator.changed,
        metadata={"positive_type": "loop_to_comprehension", "stage": "v2"},
    )


class _AccumulatorLoopToSum(_SingleBodyRewrite):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed = []

    def _rewrite_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if self.changed:
            return body
        for index in range(len(body) - 2):
            assign, loop, ret = body[index], body[index + 1], body[index + 2]
            result = _sum_replacement(assign, loop, ret)
            if result is None:
                continue
            total_name, sum_call = result
            new_body = [*body[:index], ast.Return(value=sum_call), *body[index + 3 :]]
            span = node_span(loop, self.source_code) or {}
            span.update({"kind": "accumulator_loop_to_sum", "accumulator": total_name})
            self.changed.append(span)
            return new_body
        return body


def _accumulator_loop_to_sum(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _AccumulatorLoopToSum(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="accumulator_loop_to_sum",
        role="positive",
        code=unparse(new_tree),
        confidence="likely",
        changed_spans=mutator.changed,
        metadata={"positive_type": "loop_to_builtin", "stage": "v2"},
    )


class _FirstDeMorganRewrite(ast.NodeTransformer):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed: list[dict[str, Any]] = []

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if self.changed or not isinstance(node.op, ast.Not) or not isinstance(node.operand, ast.BoolOp):
            return node
        if isinstance(node.operand.op, ast.And):
            op: ast.boolop = ast.Or()
        elif isinstance(node.operand.op, ast.Or):
            op = ast.And()
        else:
            return node
        span = node_span(node, self.source_code) or {}
        span.update({"kind": "demorgan_boolean_rewrite"})
        self.changed.append(span)
        return ast.BoolOp(op=op, values=[ast.UnaryOp(op=ast.Not(), operand=value) for value in node.operand.values])


def _demorgan_boolean_rewrite(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _FirstDeMorganRewrite(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="demorgan_boolean_rewrite",
        role="positive",
        code=unparse(new_tree),
        confidence="safe",
        changed_spans=mutator.changed,
        metadata={"positive_type": "boolean_refactor", "stage": "v2"},
    )


class _IndependentStatementBlockSwapper(_SingleBodyRewrite):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.changed = []

    def _rewrite_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if self.changed:
            return body
        for index in range(len(body) - 1):
            first, second = body[index], body[index + 1]
            if not _independent_reorderable_statements(first, second):
                continue
            new_body = list(body)
            new_body[index], new_body[index + 1] = second, first
            span = node_span(first, self.source_code) or {}
            span.update({"kind": "independent_statement_block_reorder"})
            self.changed.append(span)
            return new_body
        return body


def _swap_independent_statement_block(tree: ast.AST, code: str) -> TransformResult | None:
    new_tree = copy.deepcopy(tree)
    mutator = _IndependentStatementBlockSwapper(code)
    mutator.visit(new_tree)
    if not mutator.changed:
        return None
    return TransformResult(
        name="swap_independent_statement_block",
        role="positive",
        code=unparse(new_tree),
        confidence="likely",
        changed_spans=mutator.changed,
        metadata={"positive_type": "independent_statement_reorder", "stage": "v2"},
    )


def _bool_return_replacement(stmt: ast.If, next_stmt: ast.stmt | None) -> tuple[ast.expr, bool] | None:
    true_value: bool | None = None
    false_value: bool | None = None
    consume_next = False
    if len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Return):
        true_value = _bool_constant(stmt.body[0].value)
    if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.Return):
        false_value = _bool_constant(stmt.orelse[0].value)
    elif not stmt.orelse and isinstance(next_stmt, ast.Return):
        false_value = _bool_constant(next_stmt.value)
        consume_next = True
    if true_value is True and false_value is False:
        return ast.UnaryOp(op=ast.Not(), operand=ast.UnaryOp(op=ast.Not(), operand=stmt.test)), consume_next
    if true_value is False and false_value is True:
        return ast.UnaryOp(op=ast.Not(), operand=stmt.test), consume_next
    return None


def _bool_constant(expr: ast.expr | None) -> bool | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, bool):
        return expr.value
    return None


def _simple_independent_assignments(first: ast.stmt, second: ast.stmt) -> bool:
    if not _is_simple_local_assignment(first) or not _is_simple_local_assignment(second):
        return False
    first_stores = _store_names(first)
    second_stores = _store_names(second)
    if not first_stores or not second_stores or first_stores & second_stores:
        return False
    return not (first_stores & _load_names(second) or second_stores & _load_names(first))


def _is_simple_local_assignment(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            return False
        return _pure_expr(stmt.value)
    if isinstance(stmt, ast.AnnAssign):
        return isinstance(stmt.target, ast.Name) and stmt.value is not None and _pure_expr(stmt.value)
    return False


def _independent_reorderable_statements(first: ast.stmt, second: ast.stmt) -> bool:
    if _simple_independent_assignments(first, second):
        return True
    if not _reorderable_stmt(first) or not _reorderable_stmt(second):
        return False
    first_stores = _store_names(first)
    second_stores = _store_names(second)
    if first_stores & second_stores:
        return False
    return not (first_stores & _load_names(second) or second_stores & _load_names(first))


def _reorderable_stmt(stmt: ast.stmt) -> bool:
    if _is_simple_local_assignment(stmt):
        return True
    return isinstance(stmt, ast.Expr) and _pure_expr(stmt.value)


def _pure_expr(expr: ast.AST) -> bool:
    unsafe = (ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr, ast.Lambda)
    return not any(isinstance(node, unsafe) for node in ast.walk(expr))


def _load_names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}


def _store_names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)}


def _terminal_stmt(stmt: ast.stmt) -> bool:
    return isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue))


def _contains_node_type(nodes: list[ast.AST], node_type: type[ast.AST]) -> bool:
    return any(isinstance(item, node_type) for node in nodes for item in ast.walk(node))


def _body_assigns_name(nodes: list[ast.AST], name: str) -> bool:
    return any(name in _store_names(node) for node in nodes)


def _stable_loop_bound(expr: ast.expr) -> bool:
    return isinstance(expr, (ast.Constant, ast.Name))


def _list_append_comp_replacement(
    assign: ast.stmt, loop: ast.stmt, ret: ast.stmt
) -> tuple[str, ast.ListComp] | None:
    if not isinstance(assign, ast.Assign) or len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        return None
    out_name = assign.targets[0].id
    if not isinstance(assign.value, ast.List) or assign.value.elts:
        return None
    if not isinstance(loop, ast.For) or loop.orelse or len(loop.body) != 1:
        return None
    if not isinstance(ret, ast.Return) or not isinstance(ret.value, ast.Name) or ret.value.id != out_name:
        return None
    only = loop.body[0]
    if not isinstance(only, ast.Expr) or not isinstance(only.value, ast.Call):
        return None
    call = only.value
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "append":
        return None
    if not isinstance(call.func.value, ast.Name) or call.func.value.id != out_name:
        return None
    if len(call.args) != 1 or call.keywords or out_name in _load_names(call.args[0]):
        return None
    comp = ast.ListComp(elt=call.args[0], generators=[ast.comprehension(target=loop.target, iter=loop.iter, ifs=[], is_async=0)])
    return out_name, comp


def _sum_replacement(assign: ast.stmt, loop: ast.stmt, ret: ast.stmt) -> tuple[str, ast.Call] | None:
    if not isinstance(assign, ast.Assign) or len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        return None
    total_name = assign.targets[0].id
    if not isinstance(assign.value, ast.Constant) or assign.value.value != 0:
        return None
    if not isinstance(loop, ast.For) or loop.orelse or len(loop.body) != 1 or not isinstance(loop.target, ast.Name):
        return None
    if not isinstance(ret, ast.Return) or not isinstance(ret.value, ast.Name) or ret.value.id != total_name:
        return None
    only = loop.body[0]
    if not isinstance(only, ast.AugAssign) or not isinstance(only.target, ast.Name):
        return None
    if only.target.id != total_name or not isinstance(only.op, ast.Add):
        return None
    if not isinstance(only.value, ast.Name) or only.value.id != loop.target.id:
        return None
    return total_name, ast.Call(func=ast.Name(id="sum", ctx=ast.Load()), args=[loop.iter], keywords=[])
