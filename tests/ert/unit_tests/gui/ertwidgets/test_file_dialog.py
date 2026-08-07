from pathlib import Path

import pytest
from PyQt6 import QtCore
from pytestqt.qtbot import QtBot

from ert.gui.tools.file.file_dialog import FileDialog


def test_that_return_keeps_file_dialog_open_and_escape_closes_it(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(FileDialog, "_init_thread", lambda self: None)
    output_file = tmp_path / "output"
    output_file.touch()
    dialog = FileDialog(str(output_file), "the_step", 0, 0, 0)
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == "the_step # 0 Realization: 0 Iteration: 0"
    qtbot.keyClick(dialog, QtCore.Qt.Key.Key_Return)
    assert dialog.isVisible()
    qtbot.keyClick(dialog, QtCore.Qt.Key.Key_Escape)
    assert not dialog.isVisible()
