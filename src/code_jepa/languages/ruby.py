"""Ruby CodeSearchNet language adapter."""

from code_jepa.languages.generic import COMMON_KEYWORDS, GenericTreeSitterAdapter, LanguageConfig

RUBY_KEYWORDS = COMMON_KEYWORDS | frozenset(
    {
        "alias",
        "begin",
        "defined",
        "ensure",
        "module",
        "redo",
        "retry",
        "then",
        "undef",
        "unless",
        "until",
        "when",
    }
)


def adapter() -> GenericTreeSitterAdapter:
    return GenericTreeSitterAdapter(
        LanguageConfig(
            language="ruby",
            parser_language="ruby",
            line_comment="#",
            block_comment_start="=begin",
            block_comment_end="=end",
            keywords=RUBY_KEYWORDS,
            import_line_prefixes=("require ", "require_relative ", "include "),
            statement_suffix="",
            bool_true="true",
            bool_false="false",
            supports_ternary=True,
            not_operator="!",
            and_operator="&&",
            or_operator="||",
        )
    )
