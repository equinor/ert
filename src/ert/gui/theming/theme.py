from __future__ import annotations

from enum import Enum
from importlib.resources import files

_THEMES_SUBPATH = "resources/gui/themes"


class Theme(Enum):
    """Identifier for a visual theme shipped with the ERT GUI.

    The value is used as the base filename of the corresponding QSS file
    under ``src/ert/gui/resources/gui/themes/``.
    """

    LIGHT = "light"
    DARK = "dark"


def load_qss(theme: Theme) -> str:
    """Read the QSS file that corresponds to ``theme``.

    Files are located via :mod:`importlib.resources` under the ``ert.gui``
    package, so they are resolved correctly whether ERT is imported from a
    source checkout or an installed wheel.  Missing files raise
    ``FileNotFoundError`` — callers should treat this as a programming
    error, not a runtime condition.
    """
    resource = files("ert.gui").joinpath(f"{_THEMES_SUBPATH}/{theme.value}.qss")
    if not resource.is_file():
        raise FileNotFoundError(
            f"QSS file for theme '{theme.value}' not found at {resource}"
        )
    return resource.read_text(encoding="utf-8")
