from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt

from ert.gui.tools.plot.plot_ensemble_selection_widget import (
    EnsembleSelectCheckButton,
    EnsembleSelectionWidget,
)

from ..conftest import get_children


def test_ensemble_selection_widget_maximum_selection(qtbot: QtBot):
    test_ensemble_names = [f"case{i}" for i in range(10)]
    widget = EnsembleSelectionWidget(ensemble_names=test_ensemble_names)
    qtbot.addWidget(widget)
    buttons = get_children(widget, EnsembleSelectCheckButton, "ensemble_selector")

    # click all buttons
    for button in buttons:
        qtbot.mouseClick(button, Qt.LeftButton)

    maximum_selected = widget.MAXIMUM_SELECTED
    assert (
        sorted(widget.getPlotEnsembleNames())
        == sorted(test_ensemble_names)[:maximum_selected]
    )


def test_ensemble_selection_widget_minimum_selection(qtbot: QtBot):
    test_ensemble_names = [f"case{i}" for i in range(10)]
    widget = EnsembleSelectionWidget(ensemble_names=test_ensemble_names)
    qtbot.addWidget(widget)
    buttons = get_children(widget, EnsembleSelectCheckButton, "ensemble_selector")
    for button in buttons:
        qtbot.mouseClick(button, Qt.LeftButton)

    checked_buttons = [button for button in buttons if button.isChecked()]
    assert len(checked_buttons) > 0
    for button in checked_buttons:
        qtbot.mouseClick(button, Qt.LeftButton)
    checked_buttons = [button for button in buttons if button.isChecked()]
    assert len(checked_buttons) == widget.MINIMUM_SELECTED


def test_ensemble_selection_widget_cannot_deselect_only_active_initial_case(
    qtbot: QtBot,
):
    test_ensemble_names = [f"case{i}" for i in range(10)]
    widget = EnsembleSelectionWidget(ensemble_names=test_ensemble_names)
    qtbot.addWidget(widget)
    buttons = get_children(widget, EnsembleSelectCheckButton, "ensemble_selector")
    initial_checked_buttons = [button for button in buttons if button.isChecked()]
    assert len(initial_checked_buttons) == 1
    initial_checked_button = initial_checked_buttons[0]
    assert initial_checked_button.isChecked()
    # attempt to remove only activate case
    qtbot.mouseClick(initial_checked_button, Qt.LeftButton)
    assert initial_checked_button.isChecked()
