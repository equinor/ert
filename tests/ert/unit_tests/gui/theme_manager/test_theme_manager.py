from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QGuiApplication, QPalette
from PyQt6.QtWidgets import QApplication

from ert.gui.theme_manager import ColorSchemeManager, ColorTheme
from ert.gui.theme_manager import theme_manager as manager_module
from ert.gui.theme_manager.theme_manager import (
    _DARK_BASE_VALUE_THRESHOLD,
    detect_system_color_theme,
)


@pytest.fixture(autouse=True)
def _restore_global_stylesheet(qapp):
    original = qapp.styleSheet()
    yield
    qapp.setStyleSheet(original)


def _palette_with_base_value(value: int) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Base, QColor.fromHsv(0, 0, value))
    return palette


def _pin_system_scheme(monkeypatch, scheme: Qt.ColorScheme) -> None:
    monkeypatch.setattr(QGuiApplication.styleHints(), "colorScheme", lambda: scheme)


def _stub_qss_per_theme(monkeypatch) -> None:
    monkeypatch.setattr(
        manager_module, "process_qss", lambda theme: f"/* qss for {theme.value} */"
    )


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


def test_that_apply_raises_runtime_error_when_no_qapplication_exists(
    qtbot, monkeypatch
) -> None:
    manager = ColorSchemeManager()

    monkeypatch.setattr(
        manager_module.QApplication, "instance", staticmethod(lambda: None)
    )

    with pytest.raises(RuntimeError, match="QApplication instance not found"):
        manager.apply_stylesheet_from_qss()


def test_that_apply_sets_the_processed_qss_as_the_application_stylesheet(
    qtbot, monkeypatch
) -> None:
    manager = ColorSchemeManager()
    monkeypatch.setattr(manager_module, "process_qss", lambda _theme: "QWidget {}")

    manager.apply_stylesheet_from_qss()

    app = QApplication.instance()
    assert app is not None
    assert app.styleSheet() == "QWidget {}"


def test_that_construction_applies_the_stylesheet_for_the_detected_theme(
    qtbot, monkeypatch
) -> None:
    _pin_system_scheme(monkeypatch, Qt.ColorScheme.Dark)
    _stub_qss_per_theme(monkeypatch)

    ColorSchemeManager()

    app = QApplication.instance()
    assert app is not None
    assert app.styleSheet() == "/* qss for dark */"


def test_that_apply_keeps_previous_stylesheet_and_logs_when_qss_is_missing(
    qtbot, monkeypatch, caplog
) -> None:
    manager = ColorSchemeManager()
    app = QApplication.instance()
    assert app is not None
    previous = app.styleSheet()

    def _raise(_color_scheme: ColorTheme) -> str:
        raise FileNotFoundError("no qss")

    monkeypatch.setattr(manager_module, "process_qss", _raise)

    with caplog.at_level("ERROR"):
        manager.apply_stylesheet_from_qss()

    assert app.styleSheet() == previous
    assert "Failed to load QSS" in caplog.text


def test_that_os_theme_change_updates_current_theme_and_emits_signal(
    qtbot, monkeypatch
) -> None:
    _pin_system_scheme(monkeypatch, Qt.ColorScheme.Light)
    manager = ColorSchemeManager()
    assert manager.current_color_theme == ColorTheme.LIGHT

    received: list[ColorTheme] = []
    manager.color_theme_changed.connect(received.append)

    _pin_system_scheme(monkeypatch, Qt.ColorScheme.Dark)
    manager._on_system_scheme_changed(Qt.ColorScheme.Dark)

    assert manager.current_color_theme == ColorTheme.DARK
    assert received == [ColorTheme.DARK]


def test_that_os_theme_change_to_the_active_theme_emits_no_signal(
    qtbot, monkeypatch
) -> None:
    _pin_system_scheme(monkeypatch, Qt.ColorScheme.Dark)
    manager = ColorSchemeManager()

    received: list[ColorTheme] = []
    manager.color_theme_changed.connect(received.append)

    manager._on_system_scheme_changed(Qt.ColorScheme.Dark)

    assert received == []


def test_that_theme_change_reapplies_the_stylesheet_for_the_new_theme(
    qtbot, monkeypatch
) -> None:
    _pin_system_scheme(monkeypatch, Qt.ColorScheme.Light)
    _stub_qss_per_theme(monkeypatch)
    manager = ColorSchemeManager()
    app = QApplication.instance()
    assert app is not None
    assert app.styleSheet() == "/* qss for light */"

    _pin_system_scheme(monkeypatch, Qt.ColorScheme.Dark)
    manager._on_system_scheme_changed(Qt.ColorScheme.Dark)

    assert app.styleSheet() == "/* qss for dark */"


def test_that_style_hints_color_scheme_signal_triggers_a_stylesheet_reapply(
    qtbot, monkeypatch
) -> None:
    _pin_system_scheme(monkeypatch, Qt.ColorScheme.Light)
    _stub_qss_per_theme(monkeypatch)
    manager = ColorSchemeManager()
    app = QApplication.instance()
    assert app is not None

    _pin_system_scheme(monkeypatch, Qt.ColorScheme.Dark)
    with qtbot.waitSignal(manager.color_theme_changed, timeout=1000) as blocker:
        QGuiApplication.styleHints().colorSchemeChanged.emit(Qt.ColorScheme.Dark)

    assert blocker.args == [ColorTheme.DARK]
    assert app.styleSheet() == "/* qss for dark */"
