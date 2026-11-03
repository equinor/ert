from types import SimpleNamespace

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton

from ert.gui.summarypanel import SummaryPanel


def create_summary_panel(qtbot, steps):
    config = SimpleNamespace(
        forward_model_steps=steps,
        parameter_configurations_with_design_matrix=[],
        observation_declarations=[],
    )
    panel = SummaryPanel(config)
    qtbot.addWidget(panel)
    return panel


def test_that_forward_model_steps_are_listed_with_one_based_index_and_arguments(
    qtbot,
):
    panel = create_summary_panel(
        qtbot,
        [
            SimpleNamespace(name="MAKE_DIRECTORY", private_args={"PATH": "foo"}),
            SimpleNamespace(name="MAKE_DIRECTORY", private_args={}),
        ],
    )

    items = [
        panel.forward_model_step_list.item(row)
        for row in range(panel.forward_model_step_list.count())
    ]

    assert [item.text() for item in items] == [
        "1: MAKE_DIRECTORY(PATH=foo)",
        "2: MAKE_DIRECTORY",
    ]
    assert items[0].toolTip() == "1: MAKE_DIRECTORY(PATH=foo)"


def test_that_forward_model_steps_are_checked_by_default(qtbot):
    panel = create_summary_panel(
        qtbot,
        [
            SimpleNamespace(name="FIRST", private_args={}),
            SimpleNamespace(name="SECOND", private_args={}),
        ],
    )

    assert panel.selected_step_indices() == [1, 2]


def test_that_unchecking_all_then_inverting_selects_every_step(qtbot):
    panel = create_summary_panel(
        qtbot,
        [
            SimpleNamespace(name="FIRST", private_args={}),
            SimpleNamespace(name="SECOND", private_args={}),
        ],
    )
    panel.findChild(
        QPushButton,
        "uncheck_all_forward_model_steps",
    ).click()
    assert panel.selected_step_indices() == []

    panel.findChild(
        QPushButton,
        "invert_forward_model_steps",
    ).click()
    assert panel.selected_step_indices() == [1, 2]


def test_that_selected_step_indices_are_one_based_and_in_order(qtbot):
    panel = create_summary_panel(
        qtbot,
        [
            SimpleNamespace(name="FIRST", private_args={}),
            SimpleNamespace(name="SECOND", private_args={}),
            SimpleNamespace(name="THIRD", private_args={}),
        ],
    )
    panel.forward_model_step_list.item(1).setCheckState(Qt.CheckState.Unchecked)

    assert panel.selected_step_indices() == [1, 3]
