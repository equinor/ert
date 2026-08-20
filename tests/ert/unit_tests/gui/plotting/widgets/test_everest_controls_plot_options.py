import logging
from unittest.mock import Mock

from PyQt6.QtWidgets import QRadioButton, QToolButton

from ert.gui.plotting.utils import PlotConfig, PlotContext
from ert.gui.plotting.widgets.plot_controls.everest_controls_plot_options import (
    DEFAULT_EXPANDED_STATE,
    EverestControlsPlotOptions,
)


def test_that_everest_controls_plot_options_initializes_with_expected_default_state(
    qtbot,
):
    options = EverestControlsPlotOptions(Mock())
    qtbot.addWidget(options.get_widget())

    assert options.is_batches_selected() is True
    assert options.get_widget()._toggle_button.isChecked() is DEFAULT_EXPANDED_STATE


def test_that_the_selected_x_axis_is_written_to_the_plot_context(qtbot):
    options = EverestControlsPlotOptions(Mock())
    qtbot.addWidget(options.get_widget())
    options._display_over_controls_radio.setChecked(True)
    plot_context = PlotContext(PlotConfig(), [], [], "some_key")

    options.update_plot_context(plot_context)

    assert plot_context.by_batch is False


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
    controls_radio.click()

    connection_point.assert_called()


def test_that_selecting_x_axis_display_option_logs_sidebar_usage_once(qtbot, caplog):
    options = EverestControlsPlotOptions(Mock())
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()
    widget.findChild(QToolButton).setChecked(True)  # expand the section

    controls_radio = widget.findChild(QRadioButton, "display_over_controls_radio")
    batches_radio = widget.findChild(QRadioButton, "display_over_batches_radio")
    assert controls_radio is not None
    assert batches_radio is not None
    expected_message = "Plot sidebar option used: 'X-axis display option'"

    with caplog.at_level(logging.INFO):
        controls_radio.click()
        assert [r.getMessage() for r in caplog.records].count(expected_message) == 1

        batches_radio.click()
        assert [r.getMessage() for r in caplog.records].count(expected_message) == 1
