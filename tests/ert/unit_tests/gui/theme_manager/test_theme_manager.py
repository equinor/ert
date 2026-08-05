from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication

from ert.gui.theme_manager import ColorSchemeManager, ColorTheme
from ert.gui.theme_manager import theme_manager as manager_module


@pytest.fixture(autouse=True)
def _restore_global_stylesheet(qapp):
    original = qapp.styleSheet()
    yield
    qapp.setStyleSheet(original)


def _pin_system_scheme(monkeypatch, scheme: Qt.ColorScheme) -> None:
    monkeypatch.setattr(QGuiApplication.styleHints(), "colorScheme", lambda: scheme)


def _stub_qss_per_theme(monkeypatch) -> None:
    monkeypatch.setattr(
        manager_module, "process_qss", lambda theme: f"/* qss for {theme.value} */"
    )


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
