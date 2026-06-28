"""JavaScript CodeSearchNet language adapter."""

from code_jepa.languages.generic import COMMON_KEYWORDS, GenericTreeSitterAdapter, LanguageConfig

JAVASCRIPT_KEYWORDS = COMMON_KEYWORDS | frozenset(
    {
        "arguments",
        "delete",
        "export",
        "from",
        "get",
        "import",
        "instanceof",
        "module",
        "of",
        "set",
        "typeof",
        "undefined",
    }
)


def adapter() -> GenericTreeSitterAdapter:
    return GenericTreeSitterAdapter(
        LanguageConfig(
            language="javascript",
            parser_language="javascript",
            keywords=JAVASCRIPT_KEYWORDS,
            import_line_prefixes=("import ", "const ", "var ", "let ", "require("),
            statement_suffix=";",
            bool_true="true",
            bool_false="false",
            supports_ternary=True,
        )
    )
