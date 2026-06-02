import pandas as pd
import pytest
from matplotlib.figure import Figure

from ert.gui.tools.plot.plot_api import EnsembleObject
from ert.gui.tools.plot.plottery import PlotConfig, PlotContext
from ert.gui.tools.plot.plottery.plot_context import PlotType
from ert.gui.tools.plot.plottery.plots import EverestControlsPlot

COLOR_PALETTE_LENGTH = 8


# Copied from test_ensemble_plot.py
# Does not represent ensembles in Everest context
# Only used to be able to swap between by batch
# and by control in the tests
@pytest.fixture
def ensemble():
    return EnsembleObject(
        "ensemble_1", "id", False, "experiment_1", started_at="2012-12-10T00:00:00"
    )


@pytest.fixture
def controls_data():
    return pd.DataFrame(
        {
            "batch_id": [0, 0, 1, 1],
            "realization": [0, 0, 0, 0],
            "control_name": ["ctrl_a", "ctrl_b", "ctrl_a", "ctrl_b"],
            "control_value": [1.0, 2.0, 1.5, 2.5],
        }
    )


@pytest.fixture
def many_controls():
    names = [f"ctrl_{i}" for i in range(COLOR_PALETTE_LENGTH * 4)]
    return pd.DataFrame(
        {
            "batch_id": [0] * COLOR_PALETTE_LENGTH * 4 + [1] * COLOR_PALETTE_LENGTH * 4,
            "realization": [0] * 2 * COLOR_PALETTE_LENGTH * 4,
            "control_name": names * 2,
            "control_value": [float(i) for i in range(COLOR_PALETTE_LENGTH * 4)]
            + [float(i) + 0.5 for i in range(COLOR_PALETTE_LENGTH * 4)],
        }
    )


def test_that_plot_pr_batch_creates_one_line_per_selected_control(
    ensemble, controls_data
):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")
    context.by_batch = True

    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])

    figure = Figure()
    plot.plot(figure, context, {ensemble: controls_data}, pd.DataFrame(), {}, None)

    axes = figure.get_axes()[0]
    lines = axes.get_lines()
    # One line per selected control
    assert len(lines) == 2


def test_that_plot_pr_batch_creates_one_line_per_selected_control_with_missing_controls(
    ensemble, controls_data
):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")
    context.by_batch = True

    plot = EverestControlsPlot()
    # Select a control that is missing for batch 1
    plot.set_selected_controls(["ctrl_a", "ctrl_c"])

    figure = Figure()
    plot.plot(figure, context, {ensemble: controls_data}, pd.DataFrame(), {}, None)

    axes = figure.get_axes()[0]
    lines = axes.get_lines()
    # Still creates a line for ctrl_a, but not for ctrl_c since it's missing
    assert len(lines) == 1
    line = lines[0]
    # verify that its correct line
    assert list(line.get_xdata()) == [0, 1]
    assert list(line.get_ydata()) == [1.0, 1.5]


def test_that_no_collections_are_created_when_by_batch_is_true(ensemble, controls_data):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")
    context.by_batch = True

    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])

    figure = Figure()
    plot.plot(figure, context, {ensemble: controls_data}, pd.DataFrame(), {}, None)

    axes = figure.get_axes()[0]
    collections = axes.collections
    assert len(collections) == 0


def test_that_no_lines_are_created_when_by_batch_is_false(ensemble, controls_data):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")
    context.by_batch = False

    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])

    figure = Figure()
    plot.plot(figure, context, {ensemble: controls_data}, pd.DataFrame(), {}, None)

    axes = figure.get_axes()[0]
    lines = axes.get_lines()
    assert len(lines) == 0


def test_that_plot_pr_control_creates_one_scatter_collection_per_batch(
    ensemble, controls_data
):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")
    context.by_batch = False

    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])

    figure = Figure()
    plot.plot(figure, context, {ensemble: controls_data}, pd.DataFrame(), {}, None)

    axes = figure.get_axes()[0]
    collections = axes.collections
    # One scatter collection per batch (batch 0 and batch 1)
    assert len(collections) == 2


def test_that_plot_pr_batch_with_empty_data_creates_no_axes(ensemble):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")

    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])

    figure = Figure()
    plot.plot(figure, context, {ensemble: pd.DataFrame()}, pd.DataFrame(), {}, None)

    assert len(figure.get_axes()) == 0


def test_that_plot_pr_batch_uses_correct_x_and_y_values(ensemble, controls_data):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")
    context.by_batch = True

    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])

    figure = Figure()
    plot.plot(figure, context, {ensemble: controls_data}, pd.DataFrame(), {}, None)

    axes = figure.get_axes()[0]
    line = axes.get_lines()[0]
    # x is batch_id, y is control_value for ctrl_a
    assert list(line.get_xdata()) == [0, 1]
    assert list(line.get_ydata()) == [1.0, 1.5]


def test_that_plot_pr_control_uses_correct_x_and_y_values(ensemble, controls_data):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")
    context.by_batch = False

    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])

    figure = Figure()
    plot.plot(figure, context, {ensemble: controls_data}, pd.DataFrame(), {}, None)

    axes = figure.get_axes()[0]
    # batch 0's scatter: x=0 (index of ctrl_a), y=1.0
    offsets_0 = axes.collections[0].get_offsets()
    assert list(offsets_0[:, 0]) == [0.0]  # x positions
    assert list(offsets_0[:, 1]) == [1.0]  # y values

    # batch 1's scatter: x=0 (index of ctrl_a), y=1.5
    offsets_1 = axes.collections[1].get_offsets()
    assert list(offsets_1[:, 0]) == [0.0]
    assert list(offsets_1[:, 1]) == [1.5]


def test_that_plot_contains_legend_items_when_number_of_entries_is_below_threshold(
    ensemble, controls_data
):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")

    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])

    figure = Figure()
    plot.plot(figure, context, {ensemble: controls_data}, pd.DataFrame(), {}, None)

    legend = figure.get_axes()[0].get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert legend_texts == ["ctrl_a", "ctrl_b"]


def test_that_plot_does_not_contain_legend_when_number_of_entries_is_above_threshold(
    ensemble, many_controls
):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")

    plot = EverestControlsPlot()
    plot.set_selected_controls([f"ctrl_{i}" for i in range(20)])

    figure = Figure()
    plot.plot(figure, context, {ensemble: many_controls}, pd.DataFrame(), {}, None)

    assert len(figure.legends) == 0


def test_that_plot_rename_batch_0_in_legend_for_by_controls_plot(
    ensemble, controls_data
):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")
    context.by_batch = False

    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])

    figure = Figure()
    plot.plot(figure, context, {ensemble: controls_data}, pd.DataFrame(), {}, None)

    legend = figure.get_axes()[0].get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "initial batch" in legend_texts
    assert "batch_1" in legend_texts


def test_that_line_style_matches_when_number_of_controls_exceeds_color_palette_length(
    ensemble, many_controls
):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")
    context.by_batch = True

    plot = EverestControlsPlot()
    plot.set_selected_controls([f"ctrl_{i}" for i in range(20)])

    figure = Figure()
    plot.plot(figure, context, {ensemble: many_controls}, pd.DataFrame(), {}, None)

    axes = figure.get_axes()[0]
    lines = axes.get_lines()
    for i, line in enumerate(lines):
        if i < COLOR_PALETTE_LENGTH:
            assert line.get_linestyle() == "-"
            assert line.get_marker() == "o"
        elif i < COLOR_PALETTE_LENGTH * 2:
            assert line.get_linestyle() == "--"
            assert line.get_marker() == "o"
        elif i < COLOR_PALETTE_LENGTH * 3:
            assert line.get_linestyle() == ":"
            assert line.get_marker() == "o"
        else:
            assert line.get_linestyle() == "-."
            assert line.get_marker() == "o"


def test_that_plot_type_matches_line_when_by_batch_is_true(ensemble, controls_data):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")
    context.by_batch = True

    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])

    figure = Figure()
    plot.plot(figure, context, {ensemble: controls_data}, pd.DataFrame(), {}, None)

    assert context.plot_type == PlotType.LINE


def test_that_plot_type_matches_scatter_when_by_batch_is_false(ensemble, controls_data):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")
    context.by_batch = False

    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])

    figure = Figure()
    plot.plot(figure, context, {ensemble: controls_data}, pd.DataFrame(), {}, None)

    assert context.plot_type == PlotType.SCATTER


def test_that_date_support_is_deactivated(ensemble, controls_data):
    config = PlotConfig(title="Controls Plot")
    context = PlotContext(config, [ensemble], [1], "controls")

    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])

    figure = Figure()
    plot.plot(figure, context, {ensemble: controls_data}, pd.DataFrame(), {}, None)

    assert not context.is_date_support_active()
