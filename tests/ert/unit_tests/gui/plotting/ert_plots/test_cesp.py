import pandas as pd
import pytest
from matplotlib.figure import Figure

from ert.gui.plotting.ert_plots import CrossEnsembleStatisticsPlot
from ert.gui.plotting.plot_api import EnsembleObject
from ert.gui.plotting.utils import PlotConfig, PlotContext


@pytest.fixture
def ensemble_objects():
    return [
        EnsembleObject(
            f"iter_{i}",
            "id",
            False,
            "experiment_1",
            started_at="2012-12-10T00:00:00",
        )
        for i in range(3)
    ]


@pytest.fixture
def plot_context(ensemble_objects):
    return PlotContext(
        PlotConfig(),
        ensembles=ensemble_objects,
        ensembles_color_indexes=list(range(len(ensemble_objects))),
        key="test",
    )


@pytest.fixture
def ensemble_to_data(ensemble_objects):
    return {
        ensemble: pd.DataFrame({0: [1.0 + i, 2.0 + i, 3.0 + i]})
        for i, ensemble in enumerate(ensemble_objects)
    }


def test_that_box_and_scatter_plot_is_being_plotted_for_ensemble_data(
    plot_context, ensemble_to_data
):
    plot_context.scatter_plot = True  # box is true by default

    figure = Figure()

    CrossEnsembleStatisticsPlot().plot(
        figure,
        plot_context,
        ensemble_to_data,
        observation_data=pd.DataFrame(),
        std_dev_images={},
        obs_loc=None,
        key_def=None,
    )

    assert len(figure.axes) == 1
    axes = figure.axes[0]
    assert len(axes.collections) == len(ensemble_to_data)  # one scatter per ensemble
    assert len(axes.patches) == len(ensemble_to_data)  # one box per ensemble


def test_that_mean_gets_plotted_for_ensemble_data_when_enabled(
    plot_context, ensemble_to_data
):
    plot_context.box_plot = False
    plot_context.mean = True

    figure = Figure()

    CrossEnsembleStatisticsPlot().plot(
        figure,
        plot_context,
        ensemble_to_data,
        observation_data=pd.DataFrame(),
        std_dev_images={},
        obs_loc=None,
        key_def=None,
    )

    assert len(figure.axes) == 1
    axes = figure.axes[0]
    # one mean marker line per ensemble
    assert len(axes.get_lines()) == len(ensemble_to_data)


@pytest.mark.parametrize("enabled", [True, False])
def test_that_legend_items_for_ensemble_data_is_toggleable(
    plot_context, ensemble_to_data, enabled
):
    legend_items = [
        "Mean",
        "Median",
        "Outliers",
        "Scatter points",
        "Whiskers (10-90 %)",
    ]
    plot_context.mean = enabled
    plot_context.scatter_plot = enabled
    plot_context.box_plot = enabled
    plot_context.outliers = enabled

    figure = Figure()

    CrossEnsembleStatisticsPlot().plot(
        figure,
        plot_context,
        ensemble_to_data,
        observation_data=pd.DataFrame(),
        std_dev_images={},
        obs_loc=None,
        key_def=None,
    )

    axes = figure.axes[0]
    legend_texts = [text.get_text() for text in axes.get_legend().get_texts()]
    for item in legend_items:
        assert (item in legend_texts) == enabled


@pytest.mark.parametrize("outliers_enabled", [True, False])
def test_that_outliers_legend_is_shown_only_when_outliers_are_enabled(
    plot_context, ensemble_to_data, outliers_enabled
):
    plot_context.box_plot = True
    plot_context.outliers = outliers_enabled

    figure = Figure()

    CrossEnsembleStatisticsPlot().plot(
        figure,
        plot_context,
        ensemble_to_data,
        observation_data=pd.DataFrame(),
        std_dev_images={},
        obs_loc=None,
        key_def=None,
    )

    axes = figure.axes[0]
    legend_texts = [text.get_text() for text in axes.get_legend().get_texts()]
    assert ("Outliers" in legend_texts) == outliers_enabled
