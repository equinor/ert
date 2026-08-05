from __future__ import annotations

from unittest.mock import Mock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication

from ert.gui.theme_manager import ColorSchemeManager, ColorTheme
from ert.gui.theme_manager import theme_manager as manager_module
from ert.gui.theme_manager.qss_processing import process_qss


@pytest.fixture(autouse=True)
def _restore_global_stylesheet(qapp):
    original = qapp.styleSheet()
    yield
    qapp.setStyleSheet(original)


def test_that_manual_override_sets_current_color_scheme_and_emits_signal(qtbot) -> None:
    manager = ColorSchemeManager()
    starting_color_scheme = manager.current_color_scheme
    target = (
        ColorTheme.LIGHT
        if starting_color_scheme == ColorTheme.DARK
        else ColorTheme.DARK
    )

    received: list[ColorTheme] = []
    manager.color_theme_changed.connect(received.append)

    manager.set_color_scheme(target)

    assert manager.current_color_scheme == target
    assert manager.follows_system is False
    assert received == [target]


def test_that_setting_the_same_color_scheme_does_not_emit_signal(qtbot) -> None:
    manager = ColorSchemeManager()
    same = manager.current_color_scheme

    received: list[ColorTheme] = []
    manager.color_theme_changed.connect(received.append)

    manager.set_color_scheme(same)

    assert received == []


def test_that_setting_the_same_color_scheme_does_not_reapply_stylesheet(
    qtbot, monkeypatch
) -> None:
    manager = ColorSchemeManager()
    same = manager.current_color_scheme

    spy = Mock(wraps=manager.apply_stylesheet_from_qss)
    monkeypatch.setattr(manager, "apply_stylesheet_from_qss", spy)

    manager.set_color_scheme(same)

    assert spy.call_count == 0


def test_that_apply_installs_the_current_color_schemes_stylesheet_on_qapplication(
    qtbot,
) -> None:
    manager = ColorSchemeManager()
    manager.set_color_scheme(ColorTheme.DARK)

    app = QApplication.instance()
    assert app is not None
    assert app.styleSheet() == process_qss(ColorTheme.DARK)


def test_that_follow_system_resets_to_detected_color_scheme_and_re_enables_auto_follow(
    qtbot, monkeypatch
) -> None:
    monkeypatch.setattr(
        QGuiApplication.styleHints(),
        "colorScheme",
        lambda: Qt.ColorScheme.Dark,
    )
    manager = ColorSchemeManager()
    manager.set_color_scheme(ColorTheme.LIGHT)
    assert manager.follows_system is False

    manager.follow_system()

    assert manager.follows_system is True
    assert manager.current_color_scheme == ColorTheme.DARK


def test_that_os_scheme_change_is_ignored_when_color_scheme_is_pinned(qtbot) -> None:
    manager = ColorSchemeManager()
    manager.set_color_scheme(ColorTheme.LIGHT)

    received: list[ColorTheme] = []
    manager.color_theme_changed.connect(received.append)

    manager._on_system_scheme_changed(Qt.ColorScheme.Dark)

    assert manager.current_color_scheme == ColorTheme.LIGHT
    assert received == []


def test_that_os_scheme_change_updates_color_scheme_when_following_system(
    qtbot, monkeypatch
) -> None:
    manager = ColorSchemeManager()
    # Pin the detected scheme to Light BEFORE re-enabling follow-system, so
    # follow_system()'s immediate re-sync deterministically lands on LIGHT
    # regardless of the host OS colour scheme.
    monkeypatch.setattr(
        QGuiApplication.styleHints(),
        "colorScheme",
        lambda: Qt.ColorScheme.Light,
    )
    manager.set_color_scheme(ColorTheme.LIGHT)
    manager.follow_system()

    monkeypatch.setattr(
        QGuiApplication.styleHints(),
        "colorScheme",
        lambda: Qt.ColorScheme.Dark,
    )

    received: list[ColorTheme] = []
    manager.color_theme_changed.connect(received.append)

    manager._on_system_scheme_changed(Qt.ColorScheme.Dark)

    assert manager.current_color_scheme == ColorTheme.DARK
    assert received == [ColorTheme.DARK]


def test_that_apply_raises_runtime_error_when_no_qapplication_exists(
    qtbot, monkeypatch
) -> None:
    manager = ColorSchemeManager()

    monkeypatch.setattr(
        manager_module.QApplication, "instance", staticmethod(lambda: None)
    )

    with pytest.raises(RuntimeError, match="QApplication instance not found"):
        manager.apply_stylesheet_from_qss()


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
