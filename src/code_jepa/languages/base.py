"""Language adapter contracts and canonical transform inventory for Code-JEPA prep."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from code_jepa.transforms.python_ast import TransformResult

TRANSFORM_STAGES = ("v0", "v1", "v2")
CODESEARCHNET_LANGUAGES = ("python", "java", "javascript", "go", "php", "ruby")

CANONICAL_TRANSFORM_INVENTORY: dict[str, dict[str, tuple[str, ...]]] = {
    "v0": {
        "positive": (
            "surface_normalize",
            "remove_comments_or_docstrings",
            "rename_locals",
        ),
        "negative": (
            "flip_comparison",
            "flip_boolop",
            "swap_call_args",
            "wrong_variable",
            "flip_small_integer",
        ),
    },
    "v1": {
        "positive": (
            "swap_independent_assignments",
            "bool_return_simplify",
            "if_return_merge",
            "remove_unreachable_else",
            "import_sort_same_block",
        ),
        "negative": (
            "flip_membership_or_identity",
            "negate_condition",
            "flip_arithmetic_operator",
            "flip_subscript_index",
            "flip_default_value",
            "flip_sort_reverse",
            "remove_return_value",
            "remove_async_or_concurrency_wait",
        ),
    },
    "v2": {
        "positive": (
            "range_loop_to_while_or_equivalent",
            "append_loop_to_collection_literal",
            "accumulator_loop_to_builtin",
            "demorgan_rewrite",
            "swap_independent_statement_block",
        ),
        "negative": (
            "shift_loop_bound",
            "remove_guard_branch",
            "flip_exception_type",
            "drop_keyword_argument",
            "copy_to_alias_mutation",
            "drop_resource_context",
        ),
    },
}


@dataclass(frozen=True)
class ParseStatus:
    tree: Any | None
    parse_ok: bool
    compile_ok: bool
    error: str | None = None


@dataclass(frozen=True)
class LanguageUnit:
    unit_family: str
    unit_type: str
    qualified_name: str
    code: str
    start_line: int
    end_line: int
    imports_context: str = ""
    class_context: str = ""
    sibling_signatures: str = ""
    identifiers: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    ast_sequence: list[str] = field(default_factory=list)


class LanguageAdapter(Protocol):
    language: str
    implemented_transforms: dict[str, dict[str, set[str]]]

    def parse(self, code: str) -> ParseStatus: ...

    def extract_units(self, file_id: str, source: str, cfg: Any) -> list[LanguageUnit]: ...

    def spans(self, unit_id: str, code: str, max_spans: int) -> list[dict[str, Any]]: ...

    def positive_views_for_stage(
        self,
        code: str,
        stage: str,
        *,
        max_views: int = 16,
        only_transform: str = "",
    ) -> list[TransformResult]: ...

    def negative_views_for_stage(
        self,
        code: str,
        stage: str,
        *,
        max_views: int = 8,
        only_transform: str = "",
    ) -> list[TransformResult]: ...

    def identifiers_from_code(self, code: str) -> list[str]: ...

    def calls_from_code(self, code: str) -> list[str]: ...

    def ast_sequence_from_code(self, code: str) -> list[str]: ...


class MissingTransformImplementation(ValueError):
    pass


def stage_chain(stage: str) -> list[str]:
    if stage == "v0":
        return ["v0"]
    if stage == "v1":
        return ["v0", "v1"]
    if stage == "v2":
        return ["v0", "v1", "v2"]
    if stage == "extract-only":
        return []
    raise ValueError(f"unknown transform stage {stage!r}")


def role_transform_names(stage: str, role: str, *, only_transform: str = "") -> tuple[str, ...]:
    names = CANONICAL_TRANSFORM_INVENTORY[stage][role]
    if only_transform:
        return tuple(name for name in names if name == only_transform)
    return names


def validate_adapter_coverage(adapters: list[LanguageAdapter], stages: list[str]) -> None:
    missing: list[str] = []
    for adapter in adapters:
        for stage in stages:
            if stage == "extract-only":
                continue
            for role, expected in CANONICAL_TRANSFORM_INVENTORY[stage].items():
                actual = adapter.implemented_transforms.get(stage, {}).get(role, set())
                for name in expected:
                    if name not in actual:
                        missing.append(f"{adapter.language}:{stage}:{role}:{name}")
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise MissingTransformImplementation(f"missing transform implementations:\n{formatted}")


def canonical_inventory_jsonable() -> dict[str, dict[str, list[str]]]:
    return {
        stage: {role: list(names) for role, names in roles.items()}
        for stage, roles in CANONICAL_TRANSFORM_INVENTORY.items()
    }
