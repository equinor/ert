from ert.gui.ertwidgets.forward_model_step_selector import ForwardModelStepSelector


def test_that_repeated_steps_each_get_their_own_checkbox(qtbot):
    selector = ForwardModelStepSelector(["foo", "foo", "bar"])
    qtbot.addWidget(selector)
    assert [cb.text() for cb in selector._checkboxes] == ["foo", "foo", "bar"]


def test_that_deselect_all_and_select_all_toggle_every_step(qtbot):
    selector = ForwardModelStepSelector(["foo", "foo", "bar"])
    qtbot.addWidget(selector)
    assert selector.selected_indices() == [0, 1, 2]

    selector.deselect_all()
    assert selector.selected_indices() == []

    selector.select_all()
    assert selector.selected_indices() == [0, 1, 2]


def test_that_unchecking_one_repeated_step_leaves_the_others_checked(qtbot):
    selector = ForwardModelStepSelector(["foo", "foo"])
    qtbot.addWidget(selector)
    selector._checkboxes[0].setChecked(False)
    assert selector.selected_indices() == [1]
