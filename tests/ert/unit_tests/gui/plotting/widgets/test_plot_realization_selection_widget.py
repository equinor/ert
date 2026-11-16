from unittest.mock import Mock

from ert.gui.plotting.widgets.plot_realization_selection_widget import (
    RealizationSelectionWidget,
)


def test_that_first_realization_is_selected_after_construction_with_non_empty_list(
    qtbot,
):
    selector = RealizationSelectionWidget(["realization_1", "realization_2"])
    qtbot.addWidget(selector)

    assert selector.get_selected_realization() == "realization_1"


def test_that_no_realization_is_selected_after_construction_with_empty_list(qtbot):
    selector = RealizationSelectionWidget([])
    qtbot.addWidget(selector)

    assert selector.get_selected_realization() is None


def test_that_realization_selection_changed_signal_is_emitted_when_selection_changes(
    qtbot,
):
    selector = RealizationSelectionWidget(["realization_1", "realization_2"])
    qtbot.addWidget(selector)
    mock_slot = Mock()
    selector.realizationSelectionChanged.connect(mock_slot)

    selector._realizations_list.setCurrentRow(1)

    mock_slot.assert_called_once()
    assert selector.get_selected_realization() == "realization_2"
