import pytest
from PyQt5.QtWidgets import QMessageBox
from qtpy.QtCore import Qt, QTimer

from everest.detached import (
    context_stop_and_wait,
    wait_for_context,
)
from ieverest import IEverest
from ieverest.utils import APP_OUT_DIALOGS, app_output


@pytest.mark.ui_test
@pytest.mark.xdist_group(name="starts_everest")
def test_qt_dialogs(qtbot):
    wait_for_context()

    ieverest = IEverest()

    def handle_message_box(message):
        qtbot.waitUntil(lambda: ieverest._gui.findChild(QMessageBox) is not None)
        widget = ieverest._gui.findChild(QMessageBox)
        qtbot.mouseClick(widget.button(QMessageBox.Ok), Qt.MouseButton.LeftButton)
        assert message in widget.text()

    QTimer.singleShot(100, lambda: handle_message_box("info message"))
    app_output().info(
        "info message",
        channels=[
            APP_OUT_DIALOGS,
        ],
        force=True,
    )

    QTimer.singleShot(100, lambda: handle_message_box("critical message"))
    app_output().critical(
        "critical message",
        channels=[
            APP_OUT_DIALOGS,
        ],
        force=True,
    )

    QTimer.singleShot(100, lambda: handle_message_box("warning message"))
    app_output().warning(
        "warning message",
        channels=[
            APP_OUT_DIALOGS,
        ],
        force=True,
    )

    QTimer.singleShot(100, lambda: handle_message_box("debug message"))
    app_output().debug(
        "debug message",
        channels=[
            APP_OUT_DIALOGS,
        ],
        force=True,
    )

    context_stop_and_wait()
