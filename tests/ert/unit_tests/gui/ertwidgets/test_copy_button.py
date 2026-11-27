from typing import override

import pytest
from PyQt6.QtGui import QIcon
from pytestqt.qtbot import QtBot

from ert.gui.ertwidgets import copy_button
from ert.gui.ertwidgets.copy_button import CopyButton


class _RecordingCopyButton(CopyButton):
    @override
    def copy(self) -> None:
        self.copy_text("text")


@pytest.fixture
def loaded_icons(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    names: list[str] = []

    def load(name: str, *args: object, **kwargs: object) -> QIcon:
        names.append(name)
        return QIcon()

    monkeypatch.setattr(copy_button, "load_stateful_icon", load)
    return names


def test_that_copy_button_shows_check_icon_after_copy_and_restores_copy_icon(
    qtbot: QtBot, loaded_icons: list[str]
):
    button = _RecordingCopyButton()
    qtbot.addWidget(button)

    button.click()
    button.restore_timer.timeout.emit()

    assert loaded_icons == ["copy.svg", "check.svg", "copy.svg"]


def test_that_copy_button_icon_has_the_text_colors_of_the_style_sheet(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
):
    icon_colors: list[dict[str, str]] = []
    monkeypatch.setattr(
        copy_button,
        "load_stateful_icon",
        lambda name, **colors: icon_colors.append(colors) or QIcon(),
    )
    button = _RecordingCopyButton()
    qtbot.addWidget(button)

    button.setStyleSheet(
        "QPushButton#copy_button { color: #112233; }"
        "QPushButton#copy_button:disabled { color: #445566; }"
    )
    button.ensurePolished()

    assert icon_colors[-1] == {
        "resting": "#112233",
        "active": "#112233",
        "disabled": "#445566",
    }


def test_that_copy_button_restore_timer_fires_only_once(
    qtbot: QtBot, loaded_icons: list[str]
):
    button = _RecordingCopyButton()
    qtbot.addWidget(button)

    assert button.restore_timer.isSingleShot()
