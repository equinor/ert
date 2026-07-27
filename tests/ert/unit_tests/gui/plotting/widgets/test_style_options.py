import logging
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

from ert.gui.plotting.customization_dialog.style_chooser import (
    STYLESET_AREA,
    StyleChooser,
)
from ert.gui.plotting.utils import PlotConfig
from ert.gui.plotting.widgets.plot_controls.style_options import StyleOptions


def test_that_style_options_provide_default_history_and_observation_styles(qtbot):
    options = StyleOptions(Mock())
    qtbot.addWidget(options)

    assert options.get_default_style().marker == PlotConfig().default_style().marker
    assert options.get_history_style().marker == PlotConfig().history_style().marker
    assert (
        options.get_observations_style().marker
        == PlotConfig().observations_style().marker
    )
    observation_label = options._style_edits["observations"].findChild(QLabel)
    assert observation_label is not None
    assert observation_label.text() == "Obs."
    assert observation_label.toolTip() == "Observations"


def test_that_changing_a_style_chooser_invokes_the_update_callback(qtbot):
    connection_point = Mock()
    options = StyleOptions(connection_point)
    qtbot.addWidget(options)

    options._style_edits["default"]._style_chooser.marker_chooser.setCurrentIndex(1)

    assert options.get_default_style().marker == "x"
    connection_point.assert_called_once()


@pytest.mark.parametrize(
    ("style_name", "control_name", "option_name"),
    [
        (style_name, control_name, f"{display_name} {option}")
        for style_name, display_name in (
            ("default", "Default"),
            ("history", "History"),
            ("observations", "Observations"),
        )
        for control_name, option in (
            ("line_chooser", "line style"),
            ("thickness_spinner", "line width"),
            ("marker_chooser", "marker style"),
            ("size_spinner", "marker size"),
        )
    ],
)
def test_that_changing_each_style_control_logs_sidebar_usage_once(
    qtbot, caplog, style_name, control_name, option_name
):
    options = StyleOptions(Mock())
    qtbot.addWidget(options)
    control = getattr(options._style_edits[style_name]._style_chooser, control_name)
    expected_message = f"Plot sidebar option used: '{option_name}'"

    with caplog.at_level(logging.INFO):
        if control_name.endswith("chooser"):
            control.setCurrentIndex((control.currentIndex() + 1) % control.count())
            control.setCurrentIndex((control.currentIndex() + 1) % control.count())
        else:
            control.setValue(control.value() + control.singleStep())
            control.setValue(control.value() + control.singleStep())

    assert [record.getMessage() for record in caplog.records].count(
        expected_message
    ) == 1


@pytest.mark.parametrize(
    ("button_name", "option_name"),
    [
        ("_toggle", "Expand style options"),
        ("_reset_button", "Reset styles"),
    ],
)
def test_that_style_option_buttons_log_sidebar_usage_once(
    qtbot, caplog, button_name, option_name
):
    options = StyleOptions(Mock())
    qtbot.addWidget(options)
    button = getattr(options, button_name)
    expected_message = f"Plot sidebar option used: '{option_name}'"

    with caplog.at_level(logging.INFO):
        button.click()
        button.click()

    assert [record.getMessage() for record in caplog.records].count(
        expected_message
    ) == 1


def test_that_compact_style_chooser_uses_symbols_without_changing_full_labels(qtbot):
    full_size_chooser = StyleChooser()
    compact_chooser = StyleChooser(compact=True)
    qtbot.addWidget(full_size_chooser)
    qtbot.addWidget(compact_chooser)

    full_size_labels = [
        full_size_chooser.line_chooser.itemText(index)
        for index in range(full_size_chooser.line_chooser.count())
    ]
    compact_labels = [
        compact_chooser.line_chooser.itemText(index)
        for index in range(compact_chooser.line_chooser.count())
    ]

    assert full_size_labels == ["Off", "Solid", "Dashed", "Dotted", "Dash dotted"]
    assert compact_labels == ["Off", "-", "--", "..", "-."]

    dotted_index = compact_chooser.line_chooser.findData(":")
    assert (
        compact_chooser.line_chooser.itemData(dotted_index, Qt.ItemDataRole.ToolTipRole)
        == "Dotted"
    )
    assert (
        compact_chooser.line_chooser.itemData(
            dotted_index, Qt.ItemDataRole.AccessibleTextRole
        )
        == "Dotted"
    )

    circle_index = compact_chooser.marker_chooser.findData("o")
    assert compact_chooser.marker_chooser.itemText(circle_index) == "○"
    assert (
        compact_chooser.marker_chooser.itemData(
            circle_index, Qt.ItemDataRole.ToolTipRole
        )
        == "Circle"
    )
    assert (
        compact_chooser.marker_chooser.itemData(
            circle_index, Qt.ItemDataRole.AccessibleTextRole
        )
        == "Circle"
    )


def test_that_compact_area_style_disables_marker_selection(qtbot):
    chooser = StyleChooser(line_style_set=STYLESET_AREA, compact=True)
    qtbot.addWidget(chooser)

    chooser.line_chooser.setCurrentIndex(chooser.line_chooser.findData("#"))
    assert not chooser.marker_chooser.isEnabled()

    chooser.line_chooser.setCurrentIndex(chooser.line_chooser.findData("-"))
    assert chooser.marker_chooser.isEnabled()


def test_that_history_and_observation_styles_follow_data_availability(qtbot):
    options = StyleOptions(Mock())
    qtbot.addWidget(options)

    options.set_history_available(False)
    options.set_observations_available(False)
    assert options._style_edits["history"].isHidden()
    assert options._style_edits["observations"].isHidden()

    options.set_history_available(True)
    options.set_observations_available(True)
    assert not options._style_edits["history"].isHidden()
    assert not options._style_edits["observations"].isHidden()


def test_that_reset_restores_all_sidebar_styles(qtbot):
    connection_point = Mock()
    options = StyleOptions(connection_point)
    qtbot.addWidget(options)

    options._style_edits["default"]._style_chooser.marker_chooser.setCurrentIndex(1)
    options._style_edits["history"]._style_chooser.marker_chooser.setCurrentIndex(1)
    options._style_edits["observations"]._style_chooser.marker_chooser.setCurrentIndex(
        1
    )
    connection_point.reset_mock()

    options._reset_button.click()

    assert options.get_default_style().marker == PlotConfig().default_style().marker
    assert options.get_history_style().marker == PlotConfig().history_style().marker
    assert (
        options.get_observations_style().marker
        == PlotConfig().observations_style().marker
    )
    connection_point.assert_called_once()
