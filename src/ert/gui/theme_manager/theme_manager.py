from __future__ import annotations

import logging
from typing import cast

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QStyleHints
from PyQt6.QtWidgets import QApplication

from .qss_processing import QssProcessingError, process_qss
from .theme_utils import ColorTheme

logger = logging.getLogger(__name__)

_STYLE_HINTS_MISSING = "styleHints() unavailable; is a QGuiApplication running?"
_QAPPLICATION_MISSING = (
    "QApplication instance not found; construct a QApplication before applying "
    "a colour scheme."
)

# HSV value cutoff for a dark palette base: dark bases sit near 42, light ones near 255.
_DARK_BASE_VALUE_THRESHOLD = 70

_COLOR_THEME_MAP = {
    Qt.ColorScheme.Dark: ColorTheme.DARK,
    Qt.ColorScheme.Light: ColorTheme.LIGHT,
}


def _require_style_hints() -> QStyleHints:
    hints = QGuiApplication.styleHints()
    if hints is None:
        raise RuntimeError(_STYLE_HINTS_MISSING)
    return hints


def _application() -> QApplication | None:
    return cast(QApplication | None, QApplication.instance())


def _palette_fallback() -> ColorTheme:
    app = _application()
    if (
        app is not None
        and app.palette().base().color().value() < _DARK_BASE_VALUE_THRESHOLD
    ):
        return ColorTheme.DARK
    return ColorTheme.LIGHT


def detect_system_color_theme(hints: QStyleHints) -> ColorTheme:
    color_theme = _COLOR_THEME_MAP.get(hints.colorScheme())
    if color_theme is None:
        return _palette_fallback()
    return color_theme


class ColorSchemeManager(QObject):
    color_theme_changed = pyqtSignal(ColorTheme)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._style_hints = _require_style_hints()
        self._current_color_theme = detect_system_color_theme(self._style_hints)
        self._style_hints.colorSchemeChanged.connect(self._on_system_scheme_changed)
        self.apply_stylesheet_from_qss()

    @property
    def current_color_theme(self) -> ColorTheme:
        return self._current_color_theme

    def apply_stylesheet_from_qss(self) -> None:
        app = _application()
        if app is None:
            raise RuntimeError(_QAPPLICATION_MISSING)
        try:
            stylesheet = process_qss(self._current_color_theme)
        except (OSError, UnicodeDecodeError, ValueError, QssProcessingError):
            logger.exception(
                "Failed to load QSS for colour scheme '%s';"
                " keeping the previously applied stylesheet.",
                self._current_color_theme.value,
            )
            return
        app.setStyleSheet(stylesheet)

    def _on_system_scheme_changed(self, _scheme: Qt.ColorScheme) -> None:
        color_theme = detect_system_color_theme(self._style_hints)
        if color_theme == self._current_color_theme:
            return
        self._current_color_theme = color_theme
        self.apply_stylesheet_from_qss()
        self.color_theme_changed.emit(color_theme)
