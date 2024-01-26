from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt

from ert.gui.tools.plot.plot_case_selection_widget import (
    CaseSelectCheckButton,
    CaseSelectionWidget,
)

from ..conftest import get_children


def test_case_selection_widget_maximum_selection(qtbot: QtBot):
    test_case_names = [f"case{i}" for i in range(10)]
    widget = CaseSelectionWidget(case_names=test_case_names)
    qtbot.addWidget(widget)
    buttons = get_children(widget, CaseSelectCheckButton, "case_selector")

    # click all buttons
    for button in buttons:
        qtbot.mouseClick(button, Qt.LeftButton)

    maximum_selected = widget.MAXIMUM_SELECTED
    assert (
        sorted(widget.getPlotCaseNames()) == sorted(test_case_names)[:maximum_selected]
    )


def test_case_selection_widget_minimum_selection(qtbot: QtBot):
    test_case_names = [f"case{i}" for i in range(10)]
    widget = CaseSelectionWidget(case_names=test_case_names)
    qtbot.addWidget(widget)
    buttons = get_children(widget, CaseSelectCheckButton, "case_selector")
    for button in buttons:
        qtbot.mouseClick(button, Qt.LeftButton)

    checked_buttons = [button for button in buttons if button.isChecked()]
    assert len(checked_buttons) > 0
    for button in checked_buttons:
        qtbot.mouseClick(button, Qt.LeftButton)
    checked_buttons = [button for button in buttons if button.isChecked()]
    assert len(checked_buttons) == widget.MINIMUM_SELECTED


def test_case_selection_widget_cannot_deselect_only_active_initial_case(qtbot: QtBot):
    test_case_names = [f"case{i}" for i in range(10)]
    widget = CaseSelectionWidget(case_names=test_case_names)
    qtbot.addWidget(widget)
    buttons = get_children(widget, CaseSelectCheckButton, "case_selector")
    initial_checked_buttons = [button for button in buttons if button.isChecked()]
    assert len(initial_checked_buttons) == 1
    initial_checked_button = initial_checked_buttons[0]
    assert initial_checked_button.isChecked()
    # attempt to remove only activate case
    qtbot.mouseClick(initial_checked_button, Qt.LeftButton)
    assert initial_checked_button.isChecked()
