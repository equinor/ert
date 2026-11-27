import pytest
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QVBoxLayout

from ert.gui import eds_colors
from ert.gui.eds_theme import render_eds_stylesheet
from ert.gui.ertwidgets import StringBox
from ert.gui.ertwidgets.card import Card
from ert.gui.ertwidgets.models import TextModel
from ert.validation import ProperNameArgument


def _stringbox_in_card(qtbot, text: str) -> tuple[Card, StringBox]:
    card = Card()
    layout = QVBoxLayout(card)
    box = StringBox(TextModel(""), default_string="valid_name")
    box.setValidator(ProperNameArgument())
    layout.addWidget(box)
    qtbot.addWidget(card)
    card.show()
    box.setText(text)
    return card, box


@pytest.mark.parametrize(
    ("text", "invalid"),
    [("valid_name", False), ("not valid!", True), ("valid", False)],
)
def test_that_stringbox_is_marked_invalid_only_while_its_text_fails_validation(
    qtbot, text, invalid
):
    _, box = _stringbox_in_card(qtbot, text)

    assert bool(box.property("invalid")) is invalid


def test_that_stringbox_in_card_uses_the_eds_input_background(qtbot):
    card, box = _stringbox_in_card(qtbot, "valid_name")
    card.setStyleSheet(render_eds_stylesheet())

    pixel = box.grab().toImage().pixelColor(box.width() // 2, 4)

    assert pixel == QColor(eds_colors.INPUT_BACKGROUND)


def test_that_invalid_stringbox_uses_the_eds_error_background(qtbot):
    card, box = _stringbox_in_card(qtbot, "not valid!")
    card.setStyleSheet(render_eds_stylesheet())

    pixel = box.grab().toImage().pixelColor(box.width() // 2, 4)

    assert pixel == QColor("#ffd2d1")
