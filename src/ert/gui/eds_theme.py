from __future__ import annotations

from importlib.resources import files
from string import Template
from types import ModuleType

from PyQt6.QtCore import QDir
from PyQt6.QtWidgets import QApplication

from ert.gui import eds_colors
from ert.gui.detect_mode import is_dark_mode
from ert.gui.ertwidgets.suggestor import _colors as suggestor_colors


def _colors(*modules: ModuleType) -> dict[str, str]:
    return {
        name: value
        for module in modules
        for name, value in vars(module).items()
        if name.isupper() and isinstance(value, str)
    }


def render_eds_stylesheet() -> str:
    name = "eds_dark.qss" if is_dark_mode() else "eds_light.qss"
    template = files("ert.gui").joinpath("resources/eds", name).read_text("utf-8")
    return Template(template).substitute(_colors(eds_colors, suggestor_colors))


def apply_eds_theme(app: QApplication) -> None:
    """Style the application like the Equinor Design System.

    Dark mode only styles the suggestor and keeps the native look elsewhere.
    """
    QDir.addSearchPath("img", str(files("ert.gui").joinpath("resources/eds/icons")))
    app.setStyleSheet(render_eds_stylesheet())
