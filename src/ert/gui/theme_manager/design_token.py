from __future__ import annotations

import json
import re

from .theme_utils import ColorTheme, read_theming_resource

_HEX_COLOR = re.compile(r"^#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$")


def read_design_token_file(theme: ColorTheme) -> str:
    """Read a light or dark design token file from the themes resource directory."""
    return read_theming_resource(
        filename=f"themes/{theme.value}.json",
        resource_kind=f"{theme.value}-theme design token",
    )


def parse_design_tokens(raw: str, theme: ColorTheme) -> dict[str, str]:
    """Parse a theme JSON file into a dictionary."""
    try:
        tokens = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ValueError(
            f"Design tokens for the {theme.value}-theme are not valid JSON"
        ) from err

    if not isinstance(tokens, dict):
        raise ValueError(
            f"Design tokens for the {theme.value}-theme must be a JSON object"
        )
    return tokens


def validate_design_tokens(tokens: dict[str, str], theme: ColorTheme) -> dict[str, str]:
    """Ensure every token value is a valid hex colour."""
    invalid = [
        name
        for name, value in tokens.items()
        if not isinstance(value, str) or not _HEX_COLOR.fullmatch(value)
    ]

    if invalid:
        raise ValueError(
            "The following design tokens do not contain valid hex colors: "
            f"{', '.join(sorted(invalid))}"
        )

    return tokens


def load_tokens(theme: ColorTheme) -> dict[str, str]:
    raw = read_design_token_file(theme)
    parsed = parse_design_tokens(raw, theme)
    return validate_design_tokens(parsed, theme)
