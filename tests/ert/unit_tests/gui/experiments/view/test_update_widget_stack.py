import pytest

from ert.gui.experiments.view import UpdateWidgetStack


def test_that_added_update_is_retrievable_by_iteration(qtbot):
    widget = UpdateWidgetStack()
    qtbot.addWidget(widget)

    first = widget.add_update_widget(0)
    second = widget.add_update_widget(1)

    assert widget.get_update_widget_for_iteration(0) is first
    assert widget.get_update_widget_for_iteration(1) is second


def test_that_update_widget_raises_for_unknown_iteration(qtbot):
    widget = UpdateWidgetStack()
    qtbot.addWidget(widget)

    widget.add_update_widget(0)

    with pytest.raises(ValueError, match="Could not find UpdateWidget"):
        widget.get_update_widget_for_iteration(1)


def test_that_adding_an_update_shows_it_when_latest_iteration_is_selected(qtbot):
    widget = UpdateWidgetStack()
    qtbot.addWidget(widget)

    widget.add_update_widget(0)
    second = widget.add_update_widget(1)

    assert widget._iteration_selector.currentData() == 1
    assert widget._stack.currentWidget() is second


def test_that_adding_an_update_keeps_a_manually_selected_iteration_shown(qtbot):
    widget = UpdateWidgetStack()
    qtbot.addWidget(widget)

    first = widget.add_update_widget(0)
    widget.add_update_widget(1)
    widget._iteration_selector.setCurrentIndex(0)

    widget.add_update_widget(2)

    assert widget._iteration_selector.currentData() == 0
    assert widget._stack.currentWidget() is first


def test_that_selecting_an_iteration_shows_its_update(qtbot):
    widget = UpdateWidgetStack()
    qtbot.addWidget(widget)

    first = widget.add_update_widget(0)
    second = widget.add_update_widget(1)

    widget._iteration_selector.setCurrentIndex(0)
    assert widget._stack.currentWidget() is first

    widget._iteration_selector.setCurrentIndex(1)
    assert widget._stack.currentWidget() is second
