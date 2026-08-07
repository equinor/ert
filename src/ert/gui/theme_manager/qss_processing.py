from __future__ import annotations

import re

from .design_token import load_tokens
from .theme_utils import ColorTheme, read_theming_resource

_TOKEN_PATTERN = re.compile(r"\{\{([^{}\s]+)\}\}")
_INCLUDE_PATTERN = re.compile(
    r'^[ \t]*@include[ \t]+"([^"\n]+\.qss\.in)"[ \t]*$', re.MULTILINE
)
_INCLUDE_LIKE_PATTERN = re.compile(r"^[ \t]*@include\b.*$", re.MULTILINE)

_BASE_TEMPLATE = "base"


class QssProcessingError(Exception):
    """Raised when QSS template substitution fails."""


def read_qss_stylesheet_file(template_name: str) -> str:
    """Read a .qss.in template from the themes resource directory."""
    return read_theming_resource(
        filename=f"qss_stylesheet/{template_name}.qss.in",
        resource_kind=f"{template_name} QSS template",
    )


def _validate_include_directives(template: str) -> None:
    for match in _INCLUDE_LIKE_PATTERN.finditer(template):
        line = match.group(0)
        if not _INCLUDE_PATTERN.fullmatch(line):
            raise QssProcessingError(
                f"Malformed @include directive: {line.strip()}. "
                'Expected the form: @include "name.qss.in"'
            )


def resolve_includes(template: str, *, _seen: frozenset[str] | None = None) -> str:
    """Replace ``@include "file.qss.in"`` lines with the file's content.

    A directive must occupy a whole line, may be indented, and must reference a
    ``.qss.in`` file. Includes may nest; circular references are detected.
    """
    if _seen is None:
        _seen = frozenset()

    _validate_include_directives(template)

    def _replacer(match: re.Match[str]) -> str:
        filename = match.group(1)
        stem = filename.removesuffix(".qss.in")
        if stem in _seen:
            raise QssProcessingError(
                f"Circular @include detected: {stem} is already being processed"
            )
        content = read_qss_stylesheet_file(stem)
        return resolve_includes(content, _seen=_seen | {stem})

    return _INCLUDE_PATTERN.sub(_replacer, template)


def substitute_tokens(template: str, tokens: dict[str, str]) -> str:
    """Replace all {{token-name}} placeholders with values from the token dict."""
    for name, value in tokens.items():
        template = template.replace(f"{{{{{name}}}}}", value)
    remaining = _TOKEN_PATTERN.findall(template)
    if remaining:
        raise QssProcessingError(
            f"Template references undefined tokens: {','.join(sorted(set(remaining)))}"
        )
    return template


def process_qss(theme: ColorTheme) -> str:
    """Load tokens for the given theme and produce a fully-resolved QSS string."""
    raw = read_qss_stylesheet_file(_BASE_TEMPLATE)
    resolved = resolve_includes(raw)
    tokens = load_tokens(theme)
    return substitute_tokens(resolved, tokens)
