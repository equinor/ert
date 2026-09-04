from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication

from ert.gui.theming import Theme
from ert.gui.theming.manager import detect_system_theme


def test_that_detection_returns_dark_when_style_hints_report_dark(
    qtbot, monkeypatch
) -> None:
    monkeypatch.setattr(
        QGuiApplication.styleHints(),
        "colorScheme",
        lambda: Qt.ColorScheme.Dark,
    )
    assert detect_system_theme() == Theme.DARK


def test_that_detection_returns_light_when_style_hints_report_light(
    qtbot, monkeypatch
) -> None:
    monkeypatch.setattr(
        QGuiApplication.styleHints(),
        "colorScheme",
        lambda: Qt.ColorScheme.Light,
    )
    assert detect_system_theme() == Theme.LIGHT


def test_that_detection_falls_back_to_palette_when_scheme_is_unknown(
    qtbot, monkeypatch
) -> None:
    monkeypatch.setattr(
        QGuiApplication.styleHints(),
        "colorScheme",
        lambda: Qt.ColorScheme.Unknown,
    )
    # With the default Qt palette used in tests the base colour is white
    # (value 255), which is above ``_DARK_BASE_VALUE_THRESHOLD`` in
    # ``_palette_fallback``, so the fallback path must resolve to LIGHT.
    assert detect_system_theme() == Theme.LIGHT
