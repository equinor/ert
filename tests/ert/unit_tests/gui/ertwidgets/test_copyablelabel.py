import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget

from ert.gui.ertwidgets.copyablelabel import (
    CopyableLabel,
    strip_run_path_magic_keywords,
)


@pytest.fixture(
    params=[
        ("<b>hehe/hoho</b>", "hehe/hoho"),
        ("<b>fooo/hoho-<ITER></b>", "fooo"),
        ("<b>booo/hoho-<IENS></b>", "booo"),
    ]
)
def label_testcase(request):
    return request.param


def test_copy_clickbtn(qtbot, label_testcase):
    label_markup, label = label_testcase
    copyable_label = CopyableLabel(label_markup)
    wrapper_widget = QWidget()
    wrapper_widget.setLayout(copyable_label)

    qtbot.addWidget(wrapper_widget)
    qtbot.waitExposed(wrapper_widget)

    qtbot.mouseClick(copyable_label.copy_button, Qt.MouseButton.LeftButton)

    assert QApplication.clipboard().text() == label


@pytest.mark.parametrize(
    ("run_path", "expected"),
    [
        ("", "/"),
        ("///", "/"),
        ("simulation/mypath", "simulation/mypath"),
        ("/local/path", "/local/path"),
        ("/local/reals-iters/real-<IENS>/iter-<ITER>", "/local/reals-iters"),
        ("/local/iters/iter-<ITER>", "/local/iters"),
        ("/local/reals/real-<IENS>", "/local/reals"),
    ],
)
def test_run_path_stripped(run_path, expected):
    assert strip_run_path_magic_keywords(run_path) == expected
