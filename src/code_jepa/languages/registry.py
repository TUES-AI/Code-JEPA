"""Language adapter registry."""

from __future__ import annotations

from functools import lru_cache

from code_jepa.languages.base import CODESEARCHNET_LANGUAGES, LanguageAdapter


@lru_cache(maxsize=None)
def adapter_for_language(language: str) -> LanguageAdapter:
    name = normalize_language(language)
    if name == "python":
        from code_jepa.languages.python import PythonAdapter

        return PythonAdapter()
    if name == "java":
        from code_jepa.languages.java import adapter

        return adapter()
    if name == "javascript":
        from code_jepa.languages.javascript import adapter

        return adapter()
    if name == "go":
        from code_jepa.languages.go import adapter

        return adapter()
    if name == "php":
        from code_jepa.languages.php import adapter

        return adapter()
    if name == "ruby":
        from code_jepa.languages.ruby import adapter

        return adapter()
    raise ValueError(f"unsupported language {language!r}; expected one of {CODESEARCHNET_LANGUAGES}")


def adapters_for_languages(languages: list[str]) -> list[LanguageAdapter]:
    return [adapter_for_language(language) for language in expand_languages(languages)]


def supported_languages() -> tuple[str, ...]:
    return CODESEARCHNET_LANGUAGES


def expand_languages(languages: list[str]) -> list[str]:
    if not languages or "all" in languages:
        return list(CODESEARCHNET_LANGUAGES)
    out = []
    for language in languages:
        normalized = normalize_language(language)
        if normalized not in CODESEARCHNET_LANGUAGES:
            raise ValueError(f"unsupported language {language!r}; expected one of {CODESEARCHNET_LANGUAGES} or 'all'")
        if normalized not in out:
            out.append(normalized)
    return out


def normalize_language(language: str) -> str:
    text = str(language).strip().lower()
    aliases = {
        "js": "javascript",
        "node": "javascript",
        "golang": "go",
        "py": "python",
        "python3": "python",
    }
    return aliases.get(text, text)
