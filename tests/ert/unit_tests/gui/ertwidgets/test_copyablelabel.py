import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget

from ert.gui.ertwidgets.copyablelabel import (
    CopyableLabel,
    strip_runpath_magic_keywords,
)


@pytest.fixture(
    params=[
        ("hehe/hoho", "hehe/hoho"),
        ("fooo/hoho-<ITER>", "fooo"),
        ("booo/hoho-<IENS>", "booo"),
    ]
)
def label_testcase(request):
    return request.param


def test_copy_clickbtn(qtbot, label_testcase):
    text, label = label_testcase
    copyable_label = CopyableLabel(text)
    wrapper_widget = QWidget()
    wrapper_widget.setLayout(copyable_label)

    qtbot.addWidget(wrapper_widget)
    qtbot.waitExposed(wrapper_widget)

    qtbot.mouseClick(copyable_label.copy_button, Qt.MouseButton.LeftButton)

    assert QApplication.clipboard().text() == label


def test_that_html_in_the_text_is_shown_literally(qtbot):
    copyable_label = CopyableLabel("<i>/path/<IENS></i>")

    assert copyable_label.label.textFormat() == Qt.TextFormat.PlainText
    assert copyable_label.label.text() == "<i>/path/<IENS></i>"


@pytest.mark.parametrize(
    ("runpath", "expected"),
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
def test_runpath_stripped(runpath, expected):
    assert strip_runpath_magic_keywords(runpath) == expected
