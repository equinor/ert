import pandas as pd
import pytest
from matplotlib.figure import Figure

from ert.gui.tools.plot.plot_api import EnsembleObject
from ert.gui.tools.plot.plottery import PlotConfig, PlotContext
from ert.gui.tools.plot.plottery.plot_context import PlotType
from ert.gui.tools.plot.plottery.plots import EverestObjectiveFunctionPlot
from ert.gui.utils import LEGEND_THRESHOLD

COLOR_PALETTE_LENGTH = 8


@pytest.fixture
def ensemble():
    return EnsembleObject(
        "batch_0", "id", False, "experiment_1", started_at="2012-12-10T00:00:00"
    )


@pytest.fixture
def realization_data():
    return pd.DataFrame(
        {
            "batch_id": [0] * LEGEND_THRESHOLD + [1] * LEGEND_THRESHOLD,
            "realization": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            "objective_function_value": [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
        }
    )


@pytest.fixture
def lots_of_realization_data():
    n_realizations = LEGEND_THRESHOLD + 1
    return pd.DataFrame(
        {
            "batch_id": [0] * n_realizations + [1] * n_realizations,
            "realization": list(range(n_realizations)) * 2,
            "objective_function_value": [float(i) for i in range(n_realizations)]
            + [float(i) + 0.5 for i in range(n_realizations)],
        }
    )


def test_that_plot_type_is_set_to_line(ensemble, realization_data):
    plot_config = PlotConfig()
    plot_context = PlotContext(plot_config, [ensemble], [0], "key")

    plot = EverestObjectiveFunctionPlot()

    figure = Figure()
    plot.plot(
        figure,
        plot_context,
        {ensemble: realization_data},
        pd.DataFrame(),
        {},
        None,
        None,
    )

    assert plot_context.plot_type == PlotType.LINE


def test_that_all_realizations_are_plotted(ensemble, realization_data):
    plot_config = PlotConfig()
    plot_context = PlotContext(plot_config, [ensemble], [0], "key")

    plot = EverestObjectiveFunctionPlot()

    figure = Figure()
    plot.plot(
        figure,
        plot_context,
        {ensemble: realization_data},
        pd.DataFrame(),
        {},
        None,
        None,
    )

    assert len(figure.get_axes()[0].get_lines()) == 5


def test_that_color_is_different_for_each_realization_when_total_below_threshold(
    ensemble, realization_data
):
    plot_config = PlotConfig()
    plot_context = PlotContext(plot_config, [ensemble], [0], "key")

    plot = EverestObjectiveFunctionPlot()

    figure = Figure()
    plot.plot(
        figure,
        plot_context,
        {ensemble: realization_data},
        pd.DataFrame(),
        {},
        None,
        None,
    )

    colors = [line.get_color() for line in figure.get_axes()[0].get_lines()]
    assert len(set(colors)) == 5


def test_that_color_is_uniform_for_num_of_realizations_above_threshold(
    ensemble, lots_of_realization_data
):
    plot_config = PlotConfig()
    plot_context = PlotContext(plot_config, [ensemble], [0], "key")

    plot = EverestObjectiveFunctionPlot()

    figure = Figure()
    plot.plot(
        figure,
        plot_context,
        {ensemble: lots_of_realization_data},
        pd.DataFrame(),
        {},
        None,
        None,
    )

    colors = [line.get_color() for line in figure.get_axes()[0].get_lines()]
    assert len(set(colors)) == 1


def test_that_legend_is_shown_when_num_of_realizations_below_threshold(
    ensemble, realization_data
):
    plot_config = PlotConfig()
    plot_context = PlotContext(plot_config, [ensemble], [0], "key")

    plot = EverestObjectiveFunctionPlot()

    figure = Figure()
    plot.plot(
        figure,
        plot_context,
        {ensemble: realization_data},
        pd.DataFrame(),
        {},
        None,
        None,
    )

    legend = figure.get_axes()[0].get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 5


def test_that_line_style_is_correct(ensemble, realization_data):
    plot_config = PlotConfig()
    plot_context = PlotContext(plot_config, [ensemble], [0], "key")

    plot = EverestObjectiveFunctionPlot()

    figure = Figure()
    plot.plot(
        figure,
        plot_context,
        {ensemble: realization_data},
        pd.DataFrame(),
        {},
        None,
        None,
    )

    lines = figure.get_axes()[0].get_lines()
    for line in lines:
        assert line.get_linestyle() == "-"
        assert line.get_marker() == "o"


def test_that_legend_is_not_shown_when_num_of_realizations_above_threshold(
    ensemble, lots_of_realization_data
):
    plot_config = PlotConfig()
    plot_context = PlotContext(plot_config, [ensemble], [0], "key")

    plot = EverestObjectiveFunctionPlot()

    figure = Figure()
    plot.plot(
        figure,
        plot_context,
        {ensemble: lots_of_realization_data},
        pd.DataFrame(),
        {},
        None,
        None,
    )

    legend = figure.get_axes()[0].get_legend()
    assert legend is None
