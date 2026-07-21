from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QToolButton
from pytestqt.qtbot import QtBot

from ert.gui.sidebar import (
    COLLAPSE_SIDEBAR,
    CREATE_PLOT,
    EXPAND_SIDEBAR,
    EXPERIMENT_STATUS,
    MANAGE_EXPERIMENTS,
    NAVIGATION_ENTRIES,
    START_EXPERIMENT,
    Sidebar,
    object_name_for_entry,
)

ALL_ENTRIES = [START_EXPERIMENT, CREATE_PLOT, MANAGE_EXPERIMENTS, EXPERIMENT_STATUS]


def _collapse_button(sidebar: Sidebar) -> QToolButton:
    button = sidebar.findChild(QToolButton, "button_collapse_sidebar")
    assert isinstance(button, QToolButton)
    return button


def _button(sidebar: Sidebar, name: str) -> QToolButton:
    button = sidebar.findChild(QToolButton, object_name_for_entry(name))
    assert isinstance(button, QToolButton)
    return button


def test_that_triggering_an_action_emits_page_requested_with_its_name(qtbot: QtBot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    for name in ALL_ENTRIES:
        with qtbot.wait_signal(sidebar.page_requested) as blocker:
            sidebar.action_for(name).trigger()
        assert blocker.args == [name]


def test_that_set_current_checks_only_the_named_action(qtbot: QtBot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    sidebar.set_current(EXPERIMENT_STATUS)

    assert sidebar.action_for(EXPERIMENT_STATUS).isChecked()
    for name in ALL_ENTRIES:
        if name != EXPERIMENT_STATUS:
            assert not sidebar.action_for(name).isChecked()


def test_that_set_current_does_not_emit_page_requested(qtbot: QtBot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    emitted: list[str] = []
    sidebar.page_requested.connect(emitted.append)

    sidebar.set_current(MANAGE_EXPERIMENTS)

    assert emitted == []


def test_that_set_status_enabled_toggles_the_experiment_status_action(qtbot: QtBot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    sidebar.set_status_enabled(False)
    assert not sidebar.action_for(EXPERIMENT_STATUS).isEnabled()

    sidebar.set_status_enabled(True)
    assert sidebar.action_for(EXPERIMENT_STATUS).isEnabled()


def test_that_right_clicking_the_create_plot_button_emits_external_plot_requested(
    qtbot: QtBot,
):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    with qtbot.wait_signal(sidebar.external_plot_requested, timeout=1000):
        qtbot.mouseClick(_button(sidebar, CREATE_PLOT), Qt.MouseButton.RightButton)


def test_that_left_click_on_create_plot_button_does_not_emit_external_plot_request(
    qtbot: QtBot,
):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    emitted: list[None] = []
    sidebar.external_plot_requested.connect(lambda: emitted.append(None))

    qtbot.mouseClick(_button(sidebar, CREATE_PLOT), Qt.MouseButton.LeftButton)

    assert emitted == []


def test_that_action_for_returns_the_action_registered_for_a_name(qtbot: QtBot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    action = sidebar.action_for(CREATE_PLOT)

    assert isinstance(action, QAction)
    assert action.isCheckable()
    assert action.toolTip() == CREATE_PLOT


def test_that_object_name_for_entry_replaces_spaces_with_underscores():
    assert object_name_for_entry(START_EXPERIMENT) == "button_Start_experiment"
    assert object_name_for_entry(EXPERIMENT_STATUS) == "button_Experiment_status"


def test_that_configure_toolbar_makes_the_rail_vertical_and_immovable(qtbot: QtBot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    assert sidebar.objectName() == "sidebar"
    assert sidebar.orientation() == Qt.Orientation.Vertical
    assert sidebar.isMovable() is False
    assert sidebar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextUnderIcon


def test_that_navigation_entries_table_matches_the_created_buttons(qtbot: QtBot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    entry_names = [name for name, _icon_file in NAVIGATION_ENTRIES]

    assert entry_names == ALL_ENTRIES
    for name in entry_names:
        assert _button(sidebar, name) is not None


def test_that_create_plot_button_has_the_right_click_tooltip(qtbot: QtBot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    assert (
        _button(sidebar, CREATE_PLOT).toolTip() == "Right click to open external window"
    )


def test_that_collapse_button_is_the_first_toolbar_widget(qtbot: QtBot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    first_action = sidebar.actions()[0]
    first_button = sidebar.widgetForAction(first_action)

    assert first_button is _collapse_button(sidebar)


def test_that_sidebar_starts_expanded_with_text_under_icons(qtbot: QtBot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    assert sidebar.collapsed is False
    assert sidebar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextUnderIcon


def test_that_clicking_collapse_button_switches_to_icon_only_mode(qtbot: QtBot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    with qtbot.wait_signal(sidebar.collapsed_changed) as blocker:
        qtbot.mouseClick(_collapse_button(sidebar), Qt.MouseButton.LeftButton)

    assert blocker.args == [True]
    assert sidebar.collapsed is True
    assert sidebar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly


def test_that_clicking_collapse_button_twice_restores_text_under_icons(qtbot: QtBot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    qtbot.mouseClick(_collapse_button(sidebar), Qt.MouseButton.LeftButton)
    qtbot.mouseClick(_collapse_button(sidebar), Qt.MouseButton.LeftButton)

    assert sidebar.collapsed is False
    assert sidebar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextUnderIcon


def test_that_collapse_button_is_not_part_of_the_navigation_action_group(qtbot: QtBot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    collapse_action = sidebar.widgetForAction(sidebar.actions()[0]).defaultAction()

    assert collapse_action.actionGroup() is None
    assert collapse_action.isCheckable() is False


def test_that_collapse_button_tooltip_reflects_next_toggle_action(qtbot: QtBot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    button = _collapse_button(sidebar)
    assert button.toolTip() == COLLAPSE_SIDEBAR

    sidebar.set_collapsed(True)
    assert button.toolTip() == EXPAND_SIDEBAR

    sidebar.set_collapsed(False)
    assert button.toolTip() == COLLAPSE_SIDEBAR
