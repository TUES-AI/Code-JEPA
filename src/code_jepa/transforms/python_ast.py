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
    parsed = parse_and_compile(code)
    if parsed.tree is None:
        return []

    candidates = [
        _safe_transform(_ast_normalized, parsed.tree),
        _safe_transform(_remove_docstrings, parsed.tree),
        _safe_transform(_rename_locals, parsed.tree),
    ]
    return _dedupe_valid(code, candidates, max_views=max_views)


def hard_negative_views(code: str, *, max_views: int = 6) -> list[TransformResult]:
    parsed = parse_and_compile(code)
    if parsed.tree is None:
        return []

    candidates = [
        _safe_transform(_flip_comparison, parsed.tree, code),
        _safe_transform(_flip_boolop, parsed.tree, code),
        _safe_transform(_swap_call_args, parsed.tree, code),
        _safe_transform(_wrong_variable, parsed.tree, code),
        _safe_transform(_flip_small_integer, parsed.tree, code),
        *_harder_negative_candidates(parsed.tree, code),
    ]
    return _dedupe_valid(code, candidates, max_views=max_views)


def extra_hard_negative_views(code: str, *, max_views: int = 8) -> list[TransformResult]:
    parsed = parse_and_compile(code)
    if parsed.tree is None:
        return []
    return _dedupe_valid(code, _harder_negative_candidates(parsed.tree, code), max_views=max_views)


def _harder_negative_candidates(tree: ast.AST, code: str) -> list[TransformResult | None]:
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
