"""Go CodeSearchNet language adapter."""

from code_jepa.languages.generic import COMMON_KEYWORDS, GenericTreeSitterAdapter, LanguageConfig

GO_KEYWORDS = COMMON_KEYWORDS | frozenset(
    {
        "chan",
        "fallthrough",
        "func",
        "map",
        "range",
        "select",
        "type",
    }
)


def adapter() -> GenericTreeSitterAdapter:
    return GenericTreeSitterAdapter(
        LanguageConfig(
            language="go",
            parser_language="go",
            keywords=GO_KEYWORDS,
            import_line_prefixes=("package ", "import "),
            statement_suffix="",
            bool_true="true",
            bool_false="false",
            supports_ternary=False,
        )
    )
