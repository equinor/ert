from __future__ import annotations

from importlib.resources import files
from string import Template
from types import ModuleType

from PyQt6.QtCore import QDir
from PyQt6.QtWidgets import QApplication

from ert.gui import eds_colors
from ert.gui.ertwidgets.suggestor import _colors as suggestor_colors


def _colors(*modules: ModuleType) -> dict[str, str]:
    return {
        name: value
        for module in modules
        for name, value in vars(module).items()
        if name.isupper() and isinstance(value, str)
    }


def render_eds_stylesheet() -> str:
    template = files("ert.gui").joinpath("resources/eds/eds.qss").read_text("utf-8")
    return Template(template).substitute(_colors(eds_colors, suggestor_colors))


def apply_eds_theme(app: QApplication) -> None:
    """Style the application like the Equinor Design System.

    Rules that only apply in light mode are scoped to widgets that are only
    marked for light mode, so dark mode keeps the native look.
    """
    QDir.addSearchPath("img", str(files("ert.gui").joinpath("resources/eds/icons")))
    app.setStyleSheet(render_eds_stylesheet())
