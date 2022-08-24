import math
from os import path

import pytest
from qtpy.QtWidgets import QApplication

from ert.gui.tools.file.file_dialog import FileDialog

testCaseTextFiles = [
    "stdout-short",
    "stdout-just-right",
    "stdout-long",
    "stdout-long-and-extra-wide",
]


@pytest.mark.parametrize(
    "textFile",
    map(lambda textFile: pytest.param(textFile, id=textFile), testCaseTextFiles),
)
def test_filedialog_size(textFile, qtbot):
    filepath = path.join(
        path.dirname(path.realpath(__file__)),
        textFile,
    )
    fileDialog = FileDialog(filepath, "the-job", 42, 13, 23)
    qtbot.addWidget(fileDialog)
    with qtbot.waitExposed(fileDialog, timeout=2000):
        textField = fileDialog._view
        fontMetrics = textField.fontMetrics()
        charWidth = fontMetrics.averageCharWidth()
        screenHeight = QApplication.primaryScreen().geometry().height()
        expectedWidth = math.ceil(120 * charWidth)
        expectedHeight = math.floor(1 / 3 * screenHeight)

        # sadly we have to wait for the dialog to be sized appropriately
        qtbot.wait(10)

        assert textField.height() == pytest.approx(
            expectedHeight, abs=0.05 * screenHeight
        )
        assert textField.width() == pytest.approx(expectedWidth, abs=charWidth * 10)
        fileDialog._stop_thread()
