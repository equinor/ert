from __future__ import annotations

import re

from .design_token import load_tokens
from .theme_utils import ColorTheme, read_theming_resource

_TOKEN_PATTERN = re.compile(r"\{\{([^{}\s]+)\}\}")

_BASE_TEMPLATE = "base"


class QssProcessingError(Exception):
    """Raised when QSS template substitution fails."""


def read_qss_stylesheet_file(template_name: str) -> str:
    """Read a .qss.in template from the themes resource directory."""
    return read_theming_resource(
        filename=f"qss_stylesheet/{template_name}.qss.in",
        resource_kind=f"{template_name} QSS template",
    )


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
    tokens = load_tokens(theme)
    return substitute_tokens(raw, tokens)
