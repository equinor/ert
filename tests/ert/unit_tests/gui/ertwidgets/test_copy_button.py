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

    monkeypatch.setattr(copy_button, "load_icon", load)
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


def test_that_copy_button_icon_is_not_tinted_in_dark_mode(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(copy_button, "is_dark_mode", lambda: True)
    plain_icons: list[str] = []
    monkeypatch.setattr(
        copy_button, "load_icon", lambda name: plain_icons.append(name) or QIcon()
    )

    qtbot.addWidget(_RecordingCopyButton())

    assert plain_icons == ["copy.svg"]
