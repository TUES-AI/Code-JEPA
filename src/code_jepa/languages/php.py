"""PHP CodeSearchNet language adapter."""

from code_jepa.languages.generic import COMMON_KEYWORDS, GenericTreeSitterAdapter, LanguageConfig

PHP_KEYWORDS = COMMON_KEYWORDS | frozenset(
    {
        "array",
        "echo",
        "extends",
        "implements",
        "include",
        "include_once",
        "namespace",
        "require",
        "require_once",
        "trait",
    }
)


def adapter() -> GenericTreeSitterAdapter:
    return GenericTreeSitterAdapter(
        LanguageConfig(
            language="php",
            parser_language="php",
            keywords=PHP_KEYWORDS,
            import_line_prefixes=("<?php", "namespace ", "use ", "require", "include"),
            statement_suffix=";",
            bool_true="true",
            bool_false="false",
            supports_ternary=True,
        )
    )
