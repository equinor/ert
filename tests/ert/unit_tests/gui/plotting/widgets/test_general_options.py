import logging
from unittest.mock import Mock

import pytest
from PyQt6.QtWidgets import QCheckBox, QPushButton, QToolButton

from ert.gui.plotting.utils import PlotConfig, PlotContext
from ert.gui.plotting.utils.plot_color_palettes import PALETTES_WITH_DESCRIPTIONS
from ert.gui.plotting.widgets.collapsible_section import CollapsibleSection
from ert.gui.plotting.widgets.plot_controls.general_options import GeneralPlotOptions


def _create_plot_context() -> PlotContext:
    return PlotContext(PlotConfig(), [], [], "some_key")


def _apply_options_to_plot_context(
    options: GeneralPlotOptions,
    plot_context: PlotContext,
    *,
    history_data_available: bool = False,
    has_observations: bool = False,
    show_observations: bool = False,
    log_scale_available: bool = False,
) -> None:
    options.update_plot_context(
        plot_context,
        history_data_available=history_data_available,
        has_observations=has_observations,
        show_observations=show_observations,
        log_scale_available=log_scale_available,
    )


def test_that_general_options_has_expected_default_checkbox_states(qtbot):
    options = GeneralPlotOptions(Mock(), is_everest=False)
    qtbot.addWidget(options.get_widget())

    assert options.legend_checkbox_state is True
    assert options.grid_checkbox_state is True
    assert options.history_checkbox_state is True
    assert options.observations_checkbox_state is True
    assert options.log_checkbox_state is False


def _expand(section: CollapsibleSection) -> None:
    button = section.findChild(QToolButton)
    assert button is not None
    button.setChecked(True)


@pytest.mark.parametrize(
    "checkbox_name",
    [
        "legend_checkbox",
        "grid_checkbox",
        "history_checkbox",
        "observations_checkbox",
        "log_scale_checkbox",
    ],
)
def test_that_toggling_a_general_option_invokes_the_connection_point(
    qtbot,
    checkbox_name,
) -> None:
    connection_point = Mock()
    options = GeneralPlotOptions(connection_point, is_everest=False)
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()
    _expand(widget)

    checkbox = widget.findChild(QCheckBox, checkbox_name)
    if checkbox_name == "log_scale_checkbox":
        checkbox.setVisible(True)
    assert checkbox is not None
    assert checkbox.isVisible()

    checkbox.click()

    connection_point.assert_called_once()


@pytest.mark.parametrize(
    ("checkbox_name", "option_name"),
    [
        ("legend_checkbox", "Legend"),
        ("grid_checkbox", "Grid"),
        ("history_checkbox", "History"),
        ("observations_checkbox", "Observations"),
        ("log_scale_checkbox", "Log scale"),
    ],
)
def test_that_toggling_a_general_option_logs_sidebar_usage_once(
    qtbot,
    caplog,
    checkbox_name,
    option_name,
) -> None:
    options = GeneralPlotOptions(Mock(), is_everest=False)
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()
    _expand(widget)

    checkbox = widget.findChild(QCheckBox, checkbox_name)
    if checkbox_name == "log_scale_checkbox":
        checkbox.setVisible(True)
    assert checkbox is not None
    assert checkbox.isVisible()

    expected_message = f"Plot sidebar option used: '{option_name}'"

    with caplog.at_level(logging.INFO):
        checkbox.click()
        assert [r.getMessage() for r in caplog.records].count(expected_message) == 1

        checkbox.click()
        assert [r.getMessage() for r in caplog.records].count(expected_message) == 1


@pytest.mark.parametrize(
    ("button_name", "axis"),
    [
        ("change_x_label_button", "x"),
        ("change_y_label_button", "y"),
    ],
)
def test_that_axis_label_button_requests_edit_for_its_axis(
    qtbot,
    button_name,
    axis,
) -> None:
    options = GeneralPlotOptions(Mock(), is_everest=False)
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()
    _expand(widget)
    button = widget.findChild(QPushButton, button_name)
    assert button is not None

    requested_axis = Mock()
    options.axisLabelEditRequested.connect(requested_axis)
    button.click()

    requested_axis.assert_called_once_with(axis)


def test_that_title_button_requests_title_edit(qtbot) -> None:
    options = GeneralPlotOptions(Mock(), is_everest=False)
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()
    _expand(widget)
    button = widget.findChild(QPushButton, "change_title_button")
    assert button is not None

    title_edit_requested = Mock()
    options.titleEditRequested.connect(title_edit_requested)
    button.click()

    title_edit_requested.assert_called_once_with()


@pytest.mark.parametrize(
    ("history_visible", "observations_visible"),
    [
        pytest.param(False, False, id="neither-visible"),
        pytest.param(True, False, id="history-visible"),
        pytest.param(False, True, id="observations-visible"),
        pytest.param(True, True, id="both-visible"),
    ],
)
def test_that_history_and_observations_visibility_can_be_set(
    qtbot,
    history_visible,
    observations_visible,
):
    options = GeneralPlotOptions(Mock(), is_everest=False)
    qtbot.addWidget(options.get_widget())
    options.get_widget().show()
    _expand(options.get_widget())

    _apply_options_to_plot_context(
        options,
        _create_plot_context(),
        history_data_available=history_visible,
        has_observations=observations_visible,
        show_observations=observations_visible,
    )

    assert options._toggle_history.isVisible() is history_visible
    assert options._toggle_observations.isVisible() is observations_visible


def test_that_everest_general_options_omit_history_and_observations(qtbot):
    options = GeneralPlotOptions(Mock(), is_everest=True)
    qtbot.addWidget(options.get_widget())
    _expand(options.get_widget())
    assert not options._toggle_history.isVisible()
    assert not options._toggle_observations.isVisible()


def test_that_legend_grid_and_color_cycle_are_written_to_the_plot_config(qtbot):
    options = GeneralPlotOptions(Mock(), is_everest=False)
    qtbot.addWidget(options.get_widget())
    options._toggle_grid.setChecked(False)
    plot_context = _create_plot_context()

    _apply_options_to_plot_context(options, plot_context)

    plot_config = plot_context.plotConfig()
    assert plot_config.is_legend_enabled()
    assert not plot_config.is_grid_enabled()
    assert plot_config.line_color_cycle() == options.get_color_cycle()


@pytest.mark.parametrize(
    ("log_scale_available", "expected_log_scale"),
    [
        pytest.param(True, True, id="tab-supports-log-scale"),
        pytest.param(False, False, id="tab-does-not-support-log-scale"),
    ],
)
def test_that_log_scale_is_applied_only_when_the_current_tab_supports_it(
    qtbot, log_scale_available, expected_log_scale
):
    options = GeneralPlotOptions(Mock(), is_everest=False)
    qtbot.addWidget(options.get_widget())
    options._toggle_log_scale.setChecked(True)
    plot_context = _create_plot_context()

    _apply_options_to_plot_context(
        options, plot_context, log_scale_available=log_scale_available
    )

    assert plot_context.log_scale is expected_log_scale


def test_that_history_and_observations_stay_disabled_when_the_key_has_neither(qtbot):
    options = GeneralPlotOptions(Mock(), is_everest=False)
    qtbot.addWidget(options.get_widget())
    plot_context = _create_plot_context()

    _apply_options_to_plot_context(options, plot_context)

    assert not plot_context.plotConfig().is_history_enabled()
    assert not plot_context.plotConfig().is_observations_enabled()


def test_that_observations_stay_enabled_while_their_toggle_is_hidden(qtbot):
    options = GeneralPlotOptions(Mock(), is_everest=False)
    qtbot.addWidget(options.get_widget())
    options.get_widget().show()
    plot_context = _create_plot_context()

    _apply_options_to_plot_context(
        options, plot_context, has_observations=True, show_observations=False
    )

    assert not options._toggle_observations.isVisible()
    assert plot_context.plotConfig().is_observations_enabled()


def test_that_everest_general_options_leave_history_and_observations_untouched(qtbot):
    options = GeneralPlotOptions(Mock(), is_everest=True)
    qtbot.addWidget(options.get_widget())
    plot_context = _create_plot_context()
    expected_observations_enabled = plot_context.plotConfig().is_observations_enabled()

    _apply_options_to_plot_context(
        options, plot_context, has_observations=True, show_observations=True
    )

    assert (
        plot_context.plotConfig().is_observations_enabled()
        is expected_observations_enabled
    )


def test_that_palette_selector_returns_the_correct_color_cycle(qtbot):
    options = GeneralPlotOptions(Mock(), is_everest=False)
    qtbot.addWidget(options.get_widget())

    selector = options._color_cycle_selector
    assert (
        selector.get_color_cycle()
        == PALETTES_WITH_DESCRIPTIONS[selector.currentText()][0]
    )


def test_that_palette_selector_child_can_be_found(qtbot):
    options = GeneralPlotOptions(Mock(), is_everest=False)
    qtbot.addWidget(options.get_widget())

    selector = options.get_widget().findChild(
        type(options._color_cycle_selector), "plot_color_palette_selector"
    )
    assert selector is not None
