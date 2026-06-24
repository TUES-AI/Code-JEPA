"""Python AST helpers for code-unit extraction and span metadata."""

from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParseResult:
    tree: ast.AST | None
    parse_ok: bool
    compile_ok: bool
    error: str | None = None


def parse_and_compile(code: str) -> ParseResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ParseResult(None, False, False, f"SyntaxError: {exc}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(tree, "<code_jepa_unit>", "exec")
    except Exception as exc:  # compile can fail on malformed transformed ASTs.
        return ParseResult(tree, True, False, f"{type(exc).__name__}: {exc}")
    return ParseResult(tree, True, True, None)


def unparse(tree: ast.AST) -> str:
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).rstrip() + "\n"


def loc_bucket(loc: int) -> str:
    if loc <= 10:
        return "tiny"
    if loc <= 40:
        return "short"
    if loc <= 120:
        return "medium"
    if loc <= 250:
        return "long"
    return "huge"


def rough_token_len(code: str) -> int:
    return len(code.replace("\n", " \n ").split())


def line_count(code: str) -> int:
    return len([line for line in code.splitlines() if line.strip()])


def node_span(node: ast.AST, code: str, *, max_source_chars: int = 240) -> dict[str, Any] | None:
    if not all(hasattr(node, attr) for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset")):
        return None
    start_line = int(getattr(node, "lineno"))
    start_col = int(getattr(node, "col_offset"))
    end_line = int(getattr(node, "end_lineno"))
    end_col = int(getattr(node, "end_col_offset"))
    segment = ast.get_source_segment(code, node) or ""
    if len(segment) > max_source_chars:
        segment = segment[:max_source_chars] + "..."
    return {
        "node_type": type(node).__name__,
        "start_line": start_line,
        "start_col": start_col,
        "end_line": end_line,
        "end_col": end_col,
        "source": segment,
    }


def ast_spans(unit_id: str, code: str, *, max_source_chars: int = 240) -> list[dict[str, Any]]:
    parsed = parse_and_compile(code)
    if parsed.tree is None:
        return []
    spans: list[dict[str, Any]] = []
    for index, node in enumerate(ast.walk(parsed.tree)):
        span = node_span(node, code, max_source_chars=max_source_chars)
        if span is None:
            continue
        span.update(
            {
                "span_id": f"{unit_id}:ast:{index}",
                "unit_id": unit_id,
                "span_index": index,
            }
        )
        spans.append(span)
    return spans
