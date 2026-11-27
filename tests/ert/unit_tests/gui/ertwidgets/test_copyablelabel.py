import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QSizePolicy, QVBoxLayout, QWidget

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


def test_that_a_wrapped_label_breaks_a_long_path_without_altering_its_text(qtbot):
    path = "/aaaaaaaa/bbbbbbbb/cccccccc/dddddddd"
    copyable_label = CopyableLabel(path, wrap=True)
    wrapper_widget = QWidget()
    wrapper_widget.setLayout(copyable_label)
    qtbot.addWidget(wrapper_widget)

    label = copyable_label.label
    assert label.heightForWidth(60) > label.heightForWidth(400)
    copyable_label.copy_button.copy()
    assert QApplication.clipboard().text() == path
    assert label.text() == f"<b>{path}</b>"


def test_that_a_wrapped_label_shows_all_its_lines_when_narrower_than_its_size_hint(
    qtbot,
):
    copyable_label = CopyableLabel("/a/b/c/d/e/f/g/h/i/j/k/l", wrap=True)
    row = QWidget()
    row.setLayout(copyable_label)
    row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
    window = QWidget()
    layout = QVBoxLayout(window)
    layout.addWidget(row)
    layout.addStretch()
    qtbot.addWidget(window)
    window.resize(window.minimumSizeHint().width(), 400)
    with qtbot.waitExposed(window):
        window.show()

    label = copyable_label.label
    assert label.width() < label.sizeHint().width()
    assert row.height() >= label.heightForWidth(label.width())
