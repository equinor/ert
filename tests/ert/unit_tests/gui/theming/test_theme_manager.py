from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication

from ert.gui.theming import Theme, ThemeManager
from ert.gui.theming.theme import load_qss


def test_that_manual_override_sets_current_theme_and_emits_signal(qtbot) -> None:
    manager = ThemeManager()
    starting_theme = manager.current_theme
    target = Theme.LIGHT if starting_theme == Theme.DARK else Theme.DARK

    received: list[Theme] = []
    manager.theme_changed.connect(received.append)

    manager.set_theme(target)

    assert manager.current_theme == target
    assert manager.follows_system is False
    assert received == [target]


def test_that_setting_the_same_theme_does_not_emit_signal(qtbot) -> None:
    manager = ThemeManager()
    same = manager.current_theme

    received: list[Theme] = []
    manager.theme_changed.connect(received.append)

    manager.set_theme(same)

    assert received == []


def test_that_apply_installs_the_current_themes_stylesheet_on_qapplication(
    qtbot,
) -> None:
    manager = ThemeManager()
    manager.set_theme(Theme.DARK)

    app = QApplication.instance()
    assert app is not None
    assert app.styleSheet() == load_qss(Theme.DARK)


def test_that_follow_system_resets_to_detected_theme_and_re_enables_auto_follow(
    qtbot, monkeypatch
) -> None:
    monkeypatch.setattr(
        QGuiApplication.styleHints(),
        "colorScheme",
        lambda: Qt.ColorScheme.Dark,
    )
    manager = ThemeManager()
    manager.set_theme(Theme.LIGHT)
    assert manager.follows_system is False

    manager.follow_system()

    assert manager.follows_system is True
    assert manager.current_theme == Theme.DARK


def test_that_os_scheme_change_is_ignored_when_theme_is_pinned(qtbot) -> None:
    manager = ThemeManager()
    manager.set_theme(Theme.LIGHT)

    received: list[Theme] = []
    manager.theme_changed.connect(received.append)

    manager._on_system_scheme_changed(Qt.ColorScheme.Dark)

    assert manager.current_theme == Theme.LIGHT
    assert received == []


def test_that_os_scheme_change_updates_theme_when_following_system(
    qtbot, monkeypatch
) -> None:
    manager = ThemeManager()
    # Pin the detected scheme to Light BEFORE re-enabling follow-system, so
    # follow_system()'s immediate re-sync deterministically lands on LIGHT
    # regardless of the host OS colour scheme.
    monkeypatch.setattr(
        QGuiApplication.styleHints(),
        "colorScheme",
        lambda: Qt.ColorScheme.Light,
    )
    manager.set_theme(Theme.LIGHT)
    manager.follow_system()

    monkeypatch.setattr(
        QGuiApplication.styleHints(),
        "colorScheme",
        lambda: Qt.ColorScheme.Dark,
    )

    received: list[Theme] = []
    manager.theme_changed.connect(received.append)

    manager._on_system_scheme_changed(Qt.ColorScheme.Dark)

    assert manager.current_theme == Theme.DARK
    assert received[-1] == Theme.DARK
