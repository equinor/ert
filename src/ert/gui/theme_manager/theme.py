from __future__ import annotations

from enum import Enum
from importlib.resources import files

_THEMES_SUBPATH = "resources/gui/themes"


class ColorScheme(Enum):
    """Identifier for a visual colour scheme shipped with the ERT GUI.

    The value is used as the base filename of the corresponding QSS file
    under ``src/ert/gui/resources/gui/themes/``.
    """

    LIGHT = "light"
    DARK = "dark"


def load_qss(color_scheme: ColorScheme) -> str:
    resource = files("ert.gui").joinpath(f"{_THEMES_SUBPATH}/{color_scheme.value}.qss")
    if not resource.is_file():
        raise FileNotFoundError(
            f"QSS file for colour scheme '{color_scheme.value}' not found at {resource}"
        )
    return resource.read_text(encoding="utf-8")
