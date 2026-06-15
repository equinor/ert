import re

import pytest
from PyQt6.QtCore import Qt
from pytestqt.qtbot import QtBot

from ert.gui.plotting.plot_api import EnsembleObject
from ert.gui.plotting.widgets import (
    EnsembleSelectionWidget,
    EnsembleSelectListWidget,
)
from tests.ert.ui_tests.gui.conftest import get_child


def generate_ensemble_objects(num):
    return [
        EnsembleObject(
            name=f"case{i}",
            id="id",
            hidden=False,
            experiment_name="exp",
            started_at="2012-12-10T00:00:00",
        )
        for i in range(num)
    ]


def selected_max_equals_max_limit(
    widget: EnsembleSelectionWidget, list_widget: EnsembleSelectListWidget
):
    return (
        len(widget.get_selected_ensembles()) == list_widget.get_maximum_ensemble_limit()
    )


def selected_min_equals_min_limit(
    widget: EnsembleSelectionWidget, list_widget: EnsembleSelectListWidget
):
    return (
        len(widget.get_selected_ensembles()) == list_widget.get_minimum_ensemble_limit()
    )


# Method to ensure user interaction is working as expected
# and not just the internal state of the widget.
# select_all_ensembles() and clear_ensemble_selection()
# should only be used to reach a certain state in the test,
# not to test the functionality of selecting/deselecting all items
def iterate_and_click_all_items(
    count, list_widget: EnsembleSelectListWidget, qtbot: QtBot
):
    for index in count:
        it = list_widget.item(index)
        qtbot.mouseClick(
            list_widget.viewport(),
            Qt.MouseButton.LeftButton,
            pos=list_widget.visualItemRect(it).center(),
        )


def setup_ensemble_widget_with_ensembles(qtbot: QtBot, num_ensembles=10):
    test_ensemble_names = generate_ensemble_objects(num_ensembles)
    widget = EnsembleSelectionWidget(test_ensemble_names, 1)
    qtbot.addWidget(widget)
    list_widget = get_child(widget, EnsembleSelectListWidget, "ensemble_selector")
    return widget, list_widget


def test_that_ensemble_selection_widget_max_min_selection(qtbot: QtBot):
    widget, list_widget = setup_ensemble_widget_with_ensembles(qtbot, 10)

    assert selected_min_equals_min_limit(widget, list_widget)  # initially one selected

    qtbot.mouseClick(
        list_widget.viewport(),
        Qt.MouseButton.LeftButton,
        pos=list_widget.visualItemRect(list_widget.item(0)).center(),
    )  # deselect the only item selected

    assert selected_min_equals_min_limit(widget, list_widget)  # still one selected

    iterate_and_click_all_items(
        range(list_widget.count()), list_widget, qtbot
    )  # select 'all'
    assert selected_max_equals_max_limit(widget, list_widget)

    iterate_and_click_all_items(
        reversed(range(list_widget.count())), list_widget, qtbot
    )  # deselect 'all'
    assert selected_min_equals_min_limit(widget, list_widget)


@pytest.mark.parametrize(
    "min_limit",
    [
        pytest.param(3, id="Minimum limit of 3"),
        pytest.param(0, id="Minimum limit of 0"),
        pytest.param(5, id="Minimum limit equal to maximum limit"),
    ],
)
def test_that_ensemble_selection_widget_clear_ensemble_selection(
    qtbot: QtBot, min_limit: int
):
    widget, list_widget = setup_ensemble_widget_with_ensembles(qtbot, 10)

    list_widget.set_minimum_ensemble_limit(min_limit)
    list_widget.select_all_ensembles()
    assert selected_max_equals_max_limit(widget, list_widget)

    list_widget.clear_ensemble_selection()
    assert selected_min_equals_min_limit(widget, list_widget)


@pytest.mark.parametrize(
    "max_limit",
    [
        pytest.param(3, id="Maximum limit of 3"),
        pytest.param(1, id="Maximum limit of 1"),
        pytest.param(10, id="Maximum limit equal to number of ensembles"),
    ],
)
def test_that_ensemble_selection_widget_select_all_ensembles(
    qtbot: QtBot, max_limit: int
):
    widget, list_widget = setup_ensemble_widget_with_ensembles(qtbot, 10)

    list_widget.set_maximum_ensemble_limit(max_limit)
    list_widget.select_all_ensembles()
    assert selected_max_equals_max_limit(widget, list_widget)


def test_that_ensemble_selection_widget_select_all_with_max_above_ensemble_count(
    qtbot: QtBot,
):
    widget, list_widget = setup_ensemble_widget_with_ensembles(qtbot, 10)

    list_widget.set_maximum_ensemble_limit(
        15
    )  # Set max limit above the number of ensembles
    list_widget.select_all_ensembles()
    assert list_widget.get_maximum_ensemble_limit() == 15
    assert len(widget.get_selected_ensembles()) == 10
    assert len(list_widget.get_checked_ensembles()) == 10


def test_that_ensemble_selection_decrease_limits_and_selection(qtbot: QtBot):
    widget, list_widget = setup_ensemble_widget_with_ensembles(qtbot, 10)

    list_widget.select_all_ensembles()
    assert selected_max_equals_max_limit(widget, list_widget)

    list_widget.set_minimum_ensemble_limit(2)
    iterate_and_click_all_items(
        reversed(range(list_widget.count())), list_widget, qtbot
    )
    assert selected_min_equals_min_limit(widget, list_widget)


def test_that_ensemble_selection_reset_limits_and_selection(qtbot: QtBot):
    widget, list_widget = setup_ensemble_widget_with_ensembles(qtbot, 10)

    list_widget.set_minimum_ensemble_limit(0)
    list_widget.set_maximum_ensemble_limit(10)
    list_widget.select_all_ensembles()
    assert selected_max_equals_max_limit(widget, list_widget)

    # Reset min to default and check that it works as expected
    list_widget.reset_minimum_ensemble_limit_to_default()
    iterate_and_click_all_items(
        reversed(range(list_widget.count())), list_widget, qtbot
    )  # deselect 'all'
    assert selected_min_equals_min_limit(widget, list_widget)

    # Reset max to default and check that it works as expected
    list_widget.reset_maximum_ensemble_limit_to_default()
    iterate_and_click_all_items(
        range(list_widget.count()), list_widget, qtbot
    )  # select 'all'
    assert selected_max_equals_max_limit(widget, list_widget)


def test_that_ensemble_selection_widget_with_invalid_limits():
    widget = EnsembleSelectionWidget([], 1)

    with pytest.raises(
        ValueError,
        match=re.escape(f"Minimum selected ensembles limit ({-1}) cannot be negative"),
    ):
        widget.set_minimum_ensemble_limit(-1)

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Minimum selected ensembles limit ({6})"
            f" cannot be greater than maximum ensembles limit ({5})"
        ),
    ):
        widget.set_minimum_ensemble_limit(6)

    widget.set_minimum_ensemble_limit(0)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Maximum selected ensembles limit ({0}) cannot be less than 1"
        ),
    ):
        widget.set_maximum_ensemble_limit(0)

    widget.set_minimum_ensemble_limit(3)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Maximum selected ensembles limit ({2})"
            f" cannot be less than minimum ensembles limit ({3})"
        ),
    ):
        widget.set_maximum_ensemble_limit(2)
