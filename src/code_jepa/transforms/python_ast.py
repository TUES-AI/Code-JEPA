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
    ]
    return _dedupe_valid(code, candidates, max_views=max_views)


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
