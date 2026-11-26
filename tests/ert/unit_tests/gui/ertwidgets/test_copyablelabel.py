import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget

from ert.gui.ertwidgets.copyablelabel import (
    CopyableLabel,
    strip_runpath_magic_keywords,
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


def _copy_from_label(qtbot, text: str, *, wrap: bool) -> str:
    copyable_label = CopyableLabel(text, wrap=wrap)
    wrapper_widget = QWidget()
    wrapper_widget.setLayout(copyable_label)
    qtbot.addWidget(wrapper_widget)

    copyable_label.copy_button.copy()
    return QApplication.clipboard().text()


def test_that_copying_a_wrapped_label_removes_the_break_opportunities_after_slashes(
    qtbot,
):
    assert _copy_from_label(qtbot, "/a/b/c", wrap=True) == "/a/b/c"


def test_that_copying_an_unwrapped_label_keeps_zero_width_spaces_in_the_text(qtbot):
    text = "/a/\u200bb"

    assert _copy_from_label(qtbot, text, wrap=False) == text
