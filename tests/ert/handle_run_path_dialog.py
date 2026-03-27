from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QMessageBox
from pytestqt.qtbot import QtBot

from ert.gui.main_window import ErtMainWindow


def handle_run_path_error_dialog(gui: ErtMainWindow, qtbot: QtBot):
    mb = gui.findChild(QMessageBox, "RUN_PATH_ERROR_BOX")

    if mb is not None:
        assert mb
        assert isinstance(mb, QMessageBox)
        # Continue without deleting the runpath
        qtbot.mouseClick(mb.buttons()[0], Qt.MouseButton.LeftButton)


def handle_run_path_dialog(
    gui: ErtMainWindow,
    qtbot: QtBot,
    delete_run_path: bool = True,
    expect_error: bool = False,
):
    mb = gui.findChildren(QMessageBox, "RUN_PATH_WARNING_BOX")
    mb = mb[-1] if mb else None

    if mb is not None:
        assert mb
        assert isinstance(mb, QMessageBox)

        if delete_run_path:
            qtbot.mouseClick(mb.checkBox(), Qt.MouseButton.LeftButton)

        qtbot.mouseClick(mb.buttons()[0], Qt.MouseButton.LeftButton)
        if expect_error:
            QTimer.singleShot(1000, lambda: handle_run_path_error_dialog(gui, qtbot))
