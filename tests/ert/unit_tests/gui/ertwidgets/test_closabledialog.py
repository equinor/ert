from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton
from pytestqt.qtbot import QtBot

from ert.gui.ertwidgets import ClosableDialog


def test_that_esc_does_not_close_when_close_is_disabled(qtbot: QtBot):
    widget = QPushButton()
    dialog = ClosableDialog("test", widget, None)
    qtbot.addWidget(dialog)

    dialog.disableCloseButton()

    closed = None

    def finished():
        nonlocal closed
        closed = "finished"

    dialog.finished.connect(finished)

    qtbot.keyPress(dialog, Qt.Key.Key_Escape)

    assert closed is None

    dialog.enableCloseButton()
    qtbot.keyPress(dialog, Qt.Key.Key_Escape)

    assert closed == "finished"
