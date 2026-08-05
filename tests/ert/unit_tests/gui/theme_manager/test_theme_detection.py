from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QGuiApplication, QPalette

from ert.gui.theme_manager import ColorTheme
from ert.gui.theme_manager import theme_manager as manager_module
from ert.gui.theme_manager.theme_manager import (
    _DARK_BASE_VALUE_THRESHOLD,
    detect_system_color_theme,
)


def _palette_with_base_value(value: int) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Base, QColor.fromHsv(0, 0, value))
    return palette


@pytest.mark.parametrize(
    ("reported_scheme", "expected_theme"),
    [
        pytest.param(Qt.ColorScheme.Dark, ColorTheme.DARK, id="dark"),
        pytest.param(Qt.ColorScheme.Light, ColorTheme.LIGHT, id="light"),
    ],
)
def test_that_detection_mirrors_the_scheme_reported_by_style_hints(
    qtbot, monkeypatch, reported_scheme, expected_theme
) -> None:
    hints = QGuiApplication.styleHints()
    monkeypatch.setattr(hints, "colorScheme", lambda: reported_scheme)
    assert detect_system_color_theme(hints) == expected_theme


def test_that_detection_falls_back_to_palette_when_scheme_is_unknown(
    qtbot, monkeypatch
) -> None:
    hints = QGuiApplication.styleHints()
    monkeypatch.setattr(hints, "colorScheme", lambda: Qt.ColorScheme.Unknown)
    # The default Qt palette used in tests has a white base colour (value
    # 255), which is above _DARK_BASE_VALUE_THRESHOLD, so the fallback path
    # must resolve to LIGHT.
    assert detect_system_color_theme(hints) == ColorTheme.LIGHT


def test_that_detection_does_not_inspect_the_palette_when_style_hints_report_a_scheme(
    qtbot, monkeypatch
) -> None:
    hints = QGuiApplication.styleHints()
    monkeypatch.setattr(hints, "colorScheme", lambda: Qt.ColorScheme.Dark)

    def _fail() -> ColorTheme:
        raise AssertionError("palette fallback must not run for an explicit scheme")

    monkeypatch.setattr(manager_module, "_palette_fallback", _fail)

    assert detect_system_color_theme(hints) == ColorTheme.DARK


def test_that_require_style_hints_raises_when_style_hints_are_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        manager_module.QGuiApplication, "styleHints", staticmethod(lambda: None)
    )
    with pytest.raises(RuntimeError, match="styleHints"):
        manager_module._require_style_hints()


def test_that_palette_fallback_returns_light_when_no_qapplication_exists(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        manager_module.QApplication, "instance", staticmethod(lambda: None)
    )
    assert manager_module._palette_fallback() == ColorTheme.LIGHT


@pytest.mark.parametrize(
    ("base_value", "expected_theme"),
    [
        pytest.param(
            _DARK_BASE_VALUE_THRESHOLD - 1,
            ColorTheme.DARK,
            id="returns-dark-when-base-value-is-below-threshold",
        ),
        pytest.param(
            _DARK_BASE_VALUE_THRESHOLD + 1,
            ColorTheme.LIGHT,
            id="returns-light-when-base-value-is-above-threshold",
        ),
        pytest.param(
            _DARK_BASE_VALUE_THRESHOLD,
            ColorTheme.LIGHT,
            id="returns-light-when-base-value-equals-threshold",
        ),
    ],
)
def test_that_palette_fallback_resolves_theme_from_base_value_threshold(
    qtbot, monkeypatch, base_value, expected_theme
) -> None:
    app = manager_module.QApplication.instance()
    monkeypatch.setattr(app, "palette", lambda: _palette_with_base_value(base_value))
    assert manager_module._palette_fallback() == expected_theme
