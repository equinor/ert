from __future__ import annotations

import logging
from typing import cast

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QStyleHints
from PyQt6.QtWidgets import QApplication

from .theme import Theme, load_qss

logger = logging.getLogger(__name__)

_STYLE_HINTS_MISSING = "styleHints() unavailable; is a QGuiApplication running?"
_QAPPLICATION_MISSING = (
    "QApplication instance not found; construct a QApplication before applying a theme."
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


def detect_system_theme() -> Theme:
    """Return the OS-reported colour scheme, or fall back to a palette heuristic.

    Uses ``QGuiApplication.styleHints().colorScheme()`` on Qt 6.5+.  When the
    style hint returns ``Unknown`` (or the platform integration does not
    report a value), the base-colour brightness of the current application
    palette is inspected instead.

    A running ``QApplication`` (or at least a ``QGuiApplication``) must exist
    before calling this function.
    """
    hints = _require_style_hints()
    scheme = hints.colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return Theme.DARK
    if scheme == Qt.ColorScheme.Light:
        return Theme.LIGHT
    return _palette_fallback()


def _palette_fallback() -> Theme:
    app = cast(QApplication | None, QApplication.instance())
    if app is None:
        return Theme.LIGHT
    return (
        Theme.DARK
        if app.palette().base().color().value() < _DARK_BASE_VALUE_THRESHOLD
        else Theme.LIGHT
    )


class ThemeManager(QObject):
    theme_changed = pyqtSignal(Theme)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._follows_system_theme: bool = True
        self._current_theme: Theme = detect_system_theme()
        hints = _require_style_hints()
        hints.colorSchemeChanged.connect(self._on_system_scheme_changed)
        self.apply_stylesheet_from_qss()

    @property
    def current_theme(self) -> Theme:
        return self._current_theme

    @property
    def follows_system(self) -> bool:
        return self._follows_system_theme

    def set_theme(self, theme: Theme) -> None:
        """Pin the manager to ``theme`` and stop following the OS scheme."""
        self._follows_system_theme = False
        self._set_theme_internal(theme)

    def follow_system(self) -> None:
        """Resume following the OS colour scheme; re-syncs immediately."""
        self._follows_system_theme = True
        self._set_theme_internal(detect_system_theme())

    def apply_stylesheet_from_qss(self) -> None:
        app = cast(QApplication | None, QApplication.instance())
        if app is None:
            raise RuntimeError(_QAPPLICATION_MISSING)
        try:
            stylesheet = load_qss(self._current_theme)
        except (OSError, UnicodeDecodeError):
            logger.exception(
                "Failed to load QSS for theme '%s'; keeping previous styling.",
                self._current_theme.value,
            )
            return
        app.setStyleSheet(stylesheet)

    def _on_system_scheme_changed(self, _scheme: Qt.ColorScheme) -> None:
        if self._follows_system_theme:
            self._set_theme_internal(detect_system_theme())

    def _set_theme_internal(self, theme: Theme) -> None:
        if theme == self._current_theme:
            return
        self._current_theme = theme
        self.apply_stylesheet_from_qss()
        self.theme_changed.emit(theme)
