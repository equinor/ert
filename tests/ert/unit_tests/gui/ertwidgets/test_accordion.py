from PyQt6.QtWidgets import QLabel
from pytestqt.qtbot import QtBot

from ert.gui.ertwidgets.accordion import Accordion


def test_that_toggling_accordion_header_shows_and_hides_content_and_emits_toggled(
    qtbot: QtBot,
):
    accordion = Accordion("Options", QLabel("content"), expanded=False)
    qtbot.addWidget(accordion)
    accordion.show()
    toggled: list[bool] = []
    accordion.toggled.connect(toggled.append)
    assert accordion.content.isHidden()

    accordion.header.click()
    assert accordion.content.isVisible()

    accordion.header.click()
    assert accordion.content.isHidden()
    assert toggled == [True, False]


def test_that_accordion_is_expanded_by_default(qtbot: QtBot):
    accordion = Accordion("Options", QLabel("content"))
    qtbot.addWidget(accordion)
    accordion.show()

    assert accordion.header.isChecked()
    assert accordion.content.isVisible()
