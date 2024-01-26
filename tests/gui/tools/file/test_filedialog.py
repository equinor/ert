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
    file_dialog = FileDialog(filepath, "the-job", 42, 13, 23)
    qtbot.addWidget(file_dialog)
    text_field = file_dialog._view
    font_metrics = text_field.fontMetrics()
    char_width = font_metrics.averageCharWidth()
    screen_height = QApplication.primaryScreen().geometry().height()
    expected_width = math.ceil(120 * char_width)
    expected_height = math.floor(1 / 3 * screen_height)

    size_hint = file_dialog.sizeHint()
    assert size_hint.height() == pytest.approx(expected_height, rel=0.05)
    assert size_hint.width() == pytest.approx(expected_width, abs=char_width * 10)
    file_dialog.accept()
