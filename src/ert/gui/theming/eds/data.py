"""Access to the bundled Equinor Design System (EDS) colour tokens.

The JSON files next to this module are a local, resolved copy of the EDS colour
foundation (see :mod:`ert.gui.theming.eds.sync` for how to refresh them). This
module loads them and exposes typed lookups used by
:mod:`ert.gui.theming.tokens`.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from ert.gui.theming.theme import ColorScheme

_DATA_DIR = Path(__file__).parent


def bundled_path(color_scheme: ColorScheme) -> Path:
    """Return the bundled EDS token file path for ``color_scheme``."""
    return _DATA_DIR / f"{color_scheme.value}.json"


@cache
def _bundle(color_scheme: ColorScheme) -> dict[str, dict[str, str]]:
    path = bundled_path(color_scheme)
    if not path.is_file():
        raise FileNotFoundError(
            f"Bundled EDS tokens for '{color_scheme.value}' not found at {path}. "
            "Run: uv run python -m ert.gui.theming.eds.sync"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def semantic(color_scheme: ColorScheme, token: str) -> str:
    """Return the hex value of an EDS *semantic* (purpose) token.

    Args:
        color_scheme: The scheme to resolve against.
        token: An EDS purpose token name, e.g. ``bg-accent-fill-muted-default``.

    Returns:
        The ``#rrggbb`` value for that token in the given scheme.

    Raises:
        KeyError: If the token is not part of the EDS semantic set.
    """
    values = _bundle(color_scheme)["semantic"]
    if token not in values:
        raise KeyError(
            f"'{token}' is not an EDS semantic token. "
            "See src/ert/gui/theming/eds/*.json for the available tokens."
        )
    return values[token]


def scale(color_scheme: ColorScheme, step: str) -> str:
    """Return the hex value of a generic EDS *scale* step.

    Args:
        color_scheme: The scheme to resolve against.
        step: A scale step name, e.g. ``accent-11`` or ``neutral-1``.

    Returns:
        The ``#rrggbb`` value for that scale step in the given scheme.

    Raises:
        KeyError: If the step is not part of the EDS scales.
    """
    values = _bundle(color_scheme)["scale"]
    if step not in values:
        raise KeyError(
            f"'{step}' is not an EDS scale step. "
            "See src/ert/gui/theming/eds/*.json for the available scales."
        )
    return values[step]
