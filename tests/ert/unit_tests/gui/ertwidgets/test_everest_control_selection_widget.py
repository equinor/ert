import pytest
from PyQt6.QtCore import Qt

from ert.gui.plotting.widgets.everest_control_selection_widget import (
    EverestControlSelectionWidget,
)


@pytest.fixture
def controls_list():
    return ["control_a", "control_b", "control_c"]


def test_that_setting_controls_populates_list(qtbot, controls_list):
    widget = EverestControlSelectionWidget(controls=[])
    qtbot.addWidget(widget)

    widget.set_controls(controls_list)

    assert widget._controls_list.count() == len(controls_list)
    assert [
        widget._controls_list.item(i).text()
        for i in range(widget._controls_list.count())
    ] == controls_list
    assert widget.get_selected_controls() == []


def test_that_selecting_controls_emits_signal(qtbot, controls_list):
    widget = EverestControlSelectionWidget(controls=controls_list)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.controlSelectionChanged, timeout=1000):
        item = widget._controls_list.item(0)
        assert item is not None
        item.setSelected(True)


def test_that_setting_pinned_to_already_pinned_control_does_not_emit_signal(
    qtbot, controls_list
):
    widget = EverestControlSelectionWidget(controls=controls_list)
    qtbot.addWidget(widget)

    widget.set_pinned_control("control_b")

    with qtbot.assertNotEmitted(widget.controlSelectionChanged):
        widget.set_pinned_control("control_b")


def test_that_setting_pinned_to_none_does_not_emit_signal_and_clears_selection(
    qtbot, controls_list
):
    widget = EverestControlSelectionWidget(controls=controls_list)
    qtbot.addWidget(widget)

    widget.set_pinned_control("control_b")

    for i in range(widget._controls_list.count()):
        item = widget._controls_list.item(i)
        assert item is not None
        item.setSelected(True)

    assert widget.get_selected_controls() == controls_list
    with qtbot.assertNotEmitted(widget.controlSelectionChanged):
        widget.set_pinned_control(None)
        assert widget.get_selected_controls() == []


def test_that_setting_pinned_control_to_non_existent_control_does_not_emit_signal_and_clears_selection(  # noqa: E501
    qtbot, controls_list
):
    widget = EverestControlSelectionWidget(controls=controls_list)
    qtbot.addWidget(widget)

    widget.set_pinned_control("control_b")

    for i in range(widget._controls_list.count()):
        item = widget._controls_list.item(i)
        assert item is not None
        item.setSelected(True)

    assert widget.get_selected_controls() == controls_list
    with qtbot.assertNotEmitted(widget.controlSelectionChanged):
        widget.set_pinned_control("non_existent_control")
        assert widget.get_selected_controls() == []


def test_that_setting_pinned_control_clears_selection_of_other_controls(
    qtbot, controls_list
):
    widget = EverestControlSelectionWidget(controls=controls_list)
    qtbot.addWidget(widget)

    for i in range(widget._controls_list.count()):
        item = widget._controls_list.item(i)
        assert item is not None
        item.setSelected(True)

    assert widget.get_selected_controls() == controls_list

    widget.set_pinned_control("control_b")

    selected = widget.get_selected_controls()
    assert selected == ["control_b"]


def test_that_setting_pinned_control_selects_it_and_makes_it_unselectable(
    qtbot, controls_list
):
    widget = EverestControlSelectionWidget(controls=controls_list)
    qtbot.addWidget(widget)

    widget.set_pinned_control("control_b")

    selected = widget.get_selected_controls()
    assert selected == ["control_b"]

    matching_items = widget._controls_list.findItems(
        "control_b", Qt.MatchFlag.MatchExactly
    )
    assert len(matching_items) == 1
    control_b = matching_items[0]
    assert control_b is not None
    assert control_b.isSelected()
    control_b.setSelected(False)
    selected_after = widget.get_selected_controls()
    assert selected_after == ["control_b"]


def test_that_changing_selection_of_multiple_controls_emit_signal_and_updates_selection(
    qtbot, controls_list
):
    widget = EverestControlSelectionWidget(controls=controls_list)
    qtbot.addWidget(widget)

    with qtbot.waitSignals([widget.controlSelectionChanged] * 3, timeout=1000):
        control_a = widget._controls_list.item(0)
        control_b = widget._controls_list.item(1)
        assert control_a is not None
        assert control_b is not None
        control_a.setSelected(True)
        assert widget.get_selected_controls() == ["control_a"]
        control_b.setSelected(True)
        assert widget.get_selected_controls() == ["control_a", "control_b"]
        control_a.setSelected(False)
        assert widget.get_selected_controls() == ["control_b"]
