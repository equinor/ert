from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QGuiApplication, QPalette

from ert.gui.theming import ColorScheme
from ert.gui.theming import manager as manager_module
from ert.gui.theming.manager import (
    _DARK_BASE_VALUE_THRESHOLD,
    detect_system_color_scheme,
)


def _palette_with_base_value(value: int) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Base, QColor.fromHsv(0, 0, value))
    return palette


def test_that_detection_returns_dark_when_style_hints_report_dark(
    qtbot, monkeypatch
) -> None:
    monkeypatch.setattr(
        QGuiApplication.styleHints(),
        "colorScheme",
        lambda: Qt.ColorScheme.Dark,
    )
    assert detect_system_color_scheme() == ColorScheme.DARK


def test_that_detection_returns_light_when_style_hints_report_light(
    qtbot, monkeypatch
) -> None:
    monkeypatch.setattr(
        QGuiApplication.styleHints(),
        "colorScheme",
        lambda: Qt.ColorScheme.Light,
    )
    assert detect_system_color_scheme() == ColorScheme.LIGHT


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
    assert detect_system_color_scheme() == ColorScheme.LIGHT


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
    assert manager_module._palette_fallback() == ColorScheme.LIGHT


@pytest.mark.parametrize(
    ("base_value", "expected_scheme"),
    [
        pytest.param(
            _DARK_BASE_VALUE_THRESHOLD - 1,
            ColorScheme.DARK,
            id="returns-dark-when-base-value-is-below-threshold",
        ),
        pytest.param(
            _DARK_BASE_VALUE_THRESHOLD + 1,
            ColorScheme.LIGHT,
            id="returns-light-when-base-value-is-above-threshold",
        ),
        pytest.param(
            _DARK_BASE_VALUE_THRESHOLD,
            ColorScheme.LIGHT,
            id="returns-light-when-base-value-equals-threshold",
        ),
    ],
)
def test_that_palette_fallback_resolves_scheme_from_base_value_threshold(
    qtbot, monkeypatch, base_value, expected_scheme
) -> None:
    app = manager_module.QApplication.instance()
    monkeypatch.setattr(app, "palette", lambda: _palette_with_base_value(base_value))
    assert manager_module._palette_fallback() == expected_scheme
