import logging
from unittest.mock import Mock

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QRadioButton

from ert.gui.plotting.widgets.plot_controls.everest_controls_plot_options import (
    EverestControlsPlotOptions,
)


def test_that_everest_controls_plot_options_initializes_with_expected_default_state(
    qtbot,
):
    options = EverestControlsPlotOptions(Mock())
    qtbot.addWidget(options.get_widget())

    assert options.is_batches_selected() is True


def test_that_toggling_everest_controls_plot_options_invokes_the_connection_point(
    qtbot,
):
    connection_point = Mock()
    options = EverestControlsPlotOptions(connection_point)
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()

    controls_radio = widget.findChild(QRadioButton, "display_over_controls_radio")
    assert controls_radio is not None
    qtbot.mouseClick(controls_radio, Qt.MouseButton.LeftButton)

    connection_point.assert_called()


def test_that_selecting_x_axis_display_option_logs_sidebar_usage_once(qtbot, caplog):
    options = EverestControlsPlotOptions(Mock())
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()

    controls_radio = widget.findChild(QRadioButton, "display_over_controls_radio")
    batches_radio = widget.findChild(QRadioButton, "display_over_batches_radio")
    assert controls_radio is not None
    assert batches_radio is not None
    expected_message = "Plot sidebar option used: 'X-axis display option'"

    with caplog.at_level(logging.INFO):
        qtbot.mouseClick(controls_radio, Qt.MouseButton.LeftButton)
        assert [r.getMessage() for r in caplog.records].count(expected_message) == 1

        qtbot.mouseClick(batches_radio, Qt.MouseButton.LeftButton)
        assert [r.getMessage() for r in caplog.records].count(expected_message) == 1
