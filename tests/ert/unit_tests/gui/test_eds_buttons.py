from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QFrame, QPushButton, QVBoxLayout

from ert.gui import eds_colors
from ert.gui.eds_theme import render_eds_stylesheet


def _button_in_central_widget(qtbot) -> tuple[QFrame, QPushButton]:
    frame = QFrame()
    frame.setObjectName("central_widget")
    button = QPushButton("Edit title")
    QVBoxLayout(frame).addWidget(button)
    qtbot.addWidget(frame)
    frame.setStyleSheet(render_eds_stylesheet())
    frame.show()
    return frame, button


def test_that_push_buttons_in_the_central_widget_have_a_primary_outline(qtbot):
    _, button = _button_in_central_widget(qtbot)

    pixel = button.grab().toImage().pixelColor(button.width() // 2, 0)

    assert pixel == QColor(eds_colors.MEDIUM_GREEN)


def test_that_disabled_push_buttons_in_the_central_widget_have_a_grey_outline(
    qtbot,
):
    _, button = _button_in_central_widget(qtbot)
    button.setEnabled(False)

    pixel = button.grab().toImage().pixelColor(button.width() // 2, 0)

    assert pixel == QColor(eds_colors.HEAVY_GREY)
