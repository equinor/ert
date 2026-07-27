from PyQt6.QtWidgets import QVBoxLayout, QWidget
from pytestqt.qtbot import QtBot

from ert.gui.theming.metrics import apply_layout


def test_that_apply_layout_sets_uniform_margins_from_int(qtbot: QtBot) -> None:
    widget = QWidget()
    qtbot.addWidget(widget)
    layout = QVBoxLayout(widget)

    apply_layout(layout, margins=8, spacing=4)

    assert layout.contentsMargins().left() == 8
    assert layout.contentsMargins().top() == 8
    assert layout.contentsMargins().right() == 8
    assert layout.contentsMargins().bottom() == 8
    assert layout.spacing() == 4


def test_that_apply_layout_sets_tuple_margins_and_spacing(qtbot: QtBot) -> None:
    widget = QWidget()
    qtbot.addWidget(widget)
    layout = QVBoxLayout(widget)

    apply_layout(layout, margins=(8, 4, 8, 8), spacing=30)

    assert layout.contentsMargins().left() == 8
    assert layout.contentsMargins().top() == 4
    assert layout.contentsMargins().right() == 8
    assert layout.contentsMargins().bottom() == 8
    assert layout.spacing() == 30
