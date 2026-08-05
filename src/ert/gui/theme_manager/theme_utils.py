from __future__ import annotations

from enum import Enum
from importlib.resources import files


class ColorTheme(Enum):
    """Identifier for a visual colour theme shipped with the ERT GUI.

    The value is used as the base filename of the corresponding resources
    under ``src/ert/gui/resources/gui/``.
    """

    LIGHT = "light"
    DARK = "dark"


def read_theming_resource(*, filename: str, resource_kind: str) -> str:
    """Read a theming resource shipped under ``ert.gui.resources.gui``.

    Args:
        filename: Path of the resource relative to ``ert.gui.resources.gui``.
        resource_kind: Human readable description of the resource, used in the
            error message raised when it is missing.

    Raises:
        FileNotFoundError: If no such resource is shipped with the package.
    """
    resource = files("ert.gui.resources.gui").joinpath(filename)
    if not resource.is_file():
        raise FileNotFoundError(f"Could not find {resource_kind} at {filename}")
    return resource.read_text(encoding="utf-8")


def load_qss(color_theme: ColorTheme) -> str:
    return read_theming_resource(
        filename=f"themes/{color_theme.value}.qss",
        resource_kind=f"{color_theme.value}-theme stylesheet",
    )
