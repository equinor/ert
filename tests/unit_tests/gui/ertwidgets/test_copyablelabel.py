import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QWidget

from ert.gui.ertwidgets.copyablelabel import CopyableLabel


@pytest.fixture(
    params=[("<b>hehehehe</b>", "hehehehe"), ("<b>foooo-<ITER></b>", "foooo-<ITER>")]
)
def label_testcase(request):
    return request.param


def test_copy_clickbtn(qtbot, tmpdir, monkeypatch, label_testcase):
    label_markup, label = label_testcase
    copyable_label = CopyableLabel(label_markup)
    wrapper_widget = QWidget()
    wrapper_widget.setLayout(copyable_label)

    qtbot.addWidget(wrapper_widget)
    qtbot.waitExposed(wrapper_widget)

    qtbot.mouseClick(copyable_label.copy_button, Qt.MouseButton.LeftButton)

    assert QApplication.clipboard().text() == label
