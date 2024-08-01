from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QPushButton,
)

from ert.gui.tools.plot.plot_window import create_error_dialog


def test_pressing_copy_button_in_error_dialog(qtbot: QtBot):
    qd = create_error_dialog("hello", "world")
    qtbot.addWidget(qd)

    qtbot.mouseClick(
        qd.findChild(QPushButton, name="copy_button"), Qt.MouseButton.LeftButton
    )
    assert QApplication.clipboard().text() == "world"
