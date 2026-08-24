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


def test_that_selected_realization_is_kept_when_it_exists_in_new_realizations(
    qtbot,
):
    selector = RealizationSelectionWidget(["0", "1", "2"])
    qtbot.addWidget(selector)
    selector._realizations_list.setCurrentRow(2)

    selector.set_realizations(["0", "1", "2", "3", "4"])

    assert selector.get_selected_realization() == "2"


def test_that_first_realization_is_selected_when_previous_selection_is_missing(
    qtbot,
):
    selector = RealizationSelectionWidget(["0", "1", "2", "3", "4"])
    qtbot.addWidget(selector)
    selector._realizations_list.setCurrentRow(4)

    selector.set_realizations(["0", "1", "2"])

    assert selector.get_selected_realization() == "0"


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
