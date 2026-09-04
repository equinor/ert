from __future__ import annotations

import logging
from typing import cast

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QStyleHints
from PyQt6.QtWidgets import QApplication

from .theme import ColorScheme, load_qss

logger = logging.getLogger(__name__)

_STYLE_HINTS_MISSING = "styleHints() unavailable; is a QGuiApplication running?"
_QAPPLICATION_MISSING = (
    "QApplication instance not found; construct a QApplication before applying "
    "a colour scheme."
)

_DARK_BASE_VALUE_THRESHOLD = 70
"""HSV ``value`` cutoff (0-255) below which the palette's base colour is
considered dark enough to imply a dark theme.

Chosen empirically: typical dark-mode base colours sit well under 70
(Fusion dark ≈ 42), while light palettes are effectively white (255), so
the midpoint is not needed and a low threshold avoids misclassifying
dim-but-light themes as dark.
"""


def _require_style_hints() -> QStyleHints:
    hints = QGuiApplication.styleHints()
    if hints is None:
        raise RuntimeError(_STYLE_HINTS_MISSING)
    return hints


def detect_system_color_scheme() -> ColorScheme:

    hints = _require_style_hints()
    scheme = hints.colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return ColorScheme.DARK
    if scheme == Qt.ColorScheme.Light:
        return ColorScheme.LIGHT
    return _palette_fallback()


def _palette_fallback() -> ColorScheme:
    app = cast(QApplication | None, QApplication.instance())
    if app is None:
        return ColorScheme.LIGHT
    return (
        ColorScheme.DARK
        if app.palette().base().color().value() < _DARK_BASE_VALUE_THRESHOLD
        else ColorScheme.LIGHT
    )


class ColorSchemeManager(QObject):
    color_scheme_changed = pyqtSignal(ColorScheme)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._follows_system_color_scheme: bool = True
        self._current_color_scheme: ColorScheme = detect_system_color_scheme()
        hints = _require_style_hints()
        hints.colorSchemeChanged.connect(self._on_system_scheme_changed)
        self.apply_stylesheet_from_qss()

    @property
    def current_color_scheme(self) -> ColorScheme:
        return self._current_color_scheme

    @property
    def follows_system(self) -> bool:
        return self._follows_system_color_scheme

    def set_color_scheme(self, color_scheme: ColorScheme) -> None:
        """Pin the manager to ``color_scheme`` and stop following the OS scheme."""
        self._follows_system_color_scheme = False
        self._set_color_scheme_internal(color_scheme)

    def follow_system(self) -> None:
        """Resume following the OS colour scheme; re-syncs immediately."""
        self._follows_system_color_scheme = True
        self._set_color_scheme_internal(detect_system_color_scheme())

    def apply_stylesheet_from_qss(self) -> None:
        app = cast(QApplication | None, QApplication.instance())
        if app is None:
            raise RuntimeError(_QAPPLICATION_MISSING)
        try:
            stylesheet = load_qss(self._current_color_scheme)
        except (OSError, UnicodeDecodeError):
            logger.exception(
                "Failed to load QSS for colour scheme '%s'; keeping previous styling.",
                self._current_color_scheme.value,
            )
            return
        app.setStyleSheet(stylesheet)

    def _on_system_scheme_changed(self, _scheme: Qt.ColorScheme) -> None:
        if self._follows_system_color_scheme:
            self._set_color_scheme_internal(detect_system_color_scheme())

    def _set_color_scheme_internal(self, color_scheme: ColorScheme) -> None:
        if color_scheme == self._current_color_scheme:
            return
        self._current_color_scheme = color_scheme
        self.apply_stylesheet_from_qss()
        self.color_scheme_changed.emit(color_scheme)
