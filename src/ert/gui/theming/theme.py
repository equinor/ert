from __future__ import annotations

import re
from enum import Enum
from importlib.resources import files

_TEMPLATE_PACKAGE = "ert.gui.theming"
_TEMPLATE_NAME = "theme.qss.in"

_TOKEN = re.compile(r"@([a-z0-9-]+)")
_SCALE_STEP = re.compile(r"\A[a-z]+-\d+\Z")
_LEADING_COMMENT = re.compile(r"\A\s*/\*.*?\*/\s*", re.DOTALL)


class ColorScheme(Enum):
    LIGHT = "light"
    DARK = "dark"


_ERT_SEMANTIC_OVERRIDES: dict[ColorScheme, dict[str, str]] = {
    ColorScheme.LIGHT: {},
    ColorScheme.DARK: {
        "bg-neutral-canvas": "neutral-3",
        "bg-neutral-surface": "neutral-14",
        "bg-accent-surface": "accent-14",
        "text-success-subtle-on-emphasis": "neutral-2",
    },
}
"""ERT-local remapping of EDS semantic tokens onto other steps of the same scale.

The EDS dark background tokens are near-black (canvas ``#0b0b0b``, surface
``#202223``). Large parts of the GUI still carry legacy inline stylesheets that
assume a light canvas and hardcode dark text, black borders and pale greys, so
those widgets are effectively unreadable on the unmodified dark tokens. Until
that inline styling is removed, dark mode resolves the three background tokens
to lighter steps, preserving the canvas < surface < nav-item elevation order
while restoring contrast.

Because that lifts the whole GUI to a mid grey, the active navigation entry
needs the opposite treatment: its EDS fill (``#2c392b``) no longer reads as
"selected" against the lightened surface, so it resolves to the near-black
``neutral-2`` instead, which keeps the pale cyan label on it at high contrast.

Light mode overrides nothing and therefore resolves straight to EDS.
"""


def _read_template() -> str:
    resource = files(_TEMPLATE_PACKAGE).joinpath(_TEMPLATE_NAME)
    if not resource.is_file():
        raise FileNotFoundError(f"QSS template not found at {resource}")
    return _LEADING_COMMENT.sub("", resource.read_text(encoding="utf-8"))


def resolve_color(color_scheme: ColorScheme, token: str) -> str:
    """Resolve an EDS semantic token to the hex value ERT actually paints.

    Prefer this over :func:`ert.gui.theming.eds.semantic` anywhere a colour is
    applied outside the QSS template (icon tints, painters), so that Python-side
    colours stay in step with :data:`_ERT_SEMANTIC_OVERRIDES` instead of
    silently falling back to the unmodified EDS value.

    Args:
        color_scheme: Colour scheme to resolve the token against.
        token: EDS semantic token name, without the leading "at" sign.

    Returns:
        The token's hex value, remapped through the ERT overrides when one
        applies to this colour scheme.
    """
    from .eds import (  # noqa: PLC0415  (avoid circular import at module load)
        scale,
        semantic,
    )

    value = semantic(color_scheme, token)  # also rejects unknown token names
    override = _ERT_SEMANTIC_OVERRIDES[color_scheme].get(token)
    return scale(color_scheme, override) if override is not None else value


def load_qss(color_scheme: ColorScheme) -> str:
    from .eds import scale  # noqa: PLC0415  (avoid circular import at module load)

    template = _read_template()

    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if _SCALE_STEP.match(token):
            return scale(color_scheme, token)
        return resolve_color(color_scheme, token)

    return _TOKEN.sub(replace, template)
