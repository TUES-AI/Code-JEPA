"""Java CodeSearchNet language adapter."""

from code_jepa.languages.generic import COMMON_KEYWORDS, GenericTreeSitterAdapter, LanguageConfig

JAVA_KEYWORDS = COMMON_KEYWORDS | frozenset(
    {
        "assert",
        "boolean",
        "byte",
        "char",
        "double",
        "extends",
        "final",
        "float",
        "implements",
        "int",
        "long",
        "native",
        "short",
        "strictfp",
        "super",
        "synchronized",
        "throws",
        "transient",
        "volatile",
    }
)


def adapter() -> GenericTreeSitterAdapter:
    return GenericTreeSitterAdapter(
        LanguageConfig(
            language="java",
            parser_language="java",
            keywords=JAVA_KEYWORDS,
            import_line_prefixes=("import ", "package "),
            statement_suffix=";",
            bool_true="true",
            bool_false="false",
            supports_ternary=True,
        )
    )
