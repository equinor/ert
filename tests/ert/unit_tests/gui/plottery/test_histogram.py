from unittest.mock import ANY, Mock

import pandas as pd
import pytest
from matplotlib.figure import Figure

import ert
from ert.gui.tools.plot.plot_api import EnsembleObject
from ert.gui.tools.plot.plottery import PlotConfig, PlotContext
from ert.gui.tools.plot.plottery.plots import HistogramPlot


@pytest.fixture(
    params=[
        pytest.param(
            (
                [
                    EnsembleObject(
                        "ensemble_1",
                        "id",
                        False,
                        "experiment_1",
                        started_at="2012-12-10T00:00:00",
                    )
                ],
                [1],
            ),
        ),
        pytest.param(([], []), id="no_ensembles"),
    ]
)
def plot_context(request):
    context = Mock(spec=PlotContext)
    context.ensembles.return_value = request.param[0]
    context.ensembles_color_indexes.return_value = request.param[1]
    title = "" + f"num_ensembles={len(request.param[0])}"
    context.plotConfig.return_value = PlotConfig(title=title)
    return context


@pytest.fixture(
    params=[
        pytest.param(pd.DataFrame([[0.1], [0.2], [0.3], [0.4], [0.5]]), id="float"),
        pytest.param(pd.DataFrame(), id="empty"),
        pytest.param(
            pd.DataFrame(["cat", "cat", "cat", "dog"] + ["fish"] * 10),
            id="categorical",
        ),
    ]
)
def ensemble_to_data_map(request, plot_context):
    if len(plot_context.ensembles()) == 0 and not request.param.empty:
        # Only test with empty ensemble list once
        pytest.skip()
    if not request.param.empty and request.param[0].dtype == "object":
        # categorial and logscale is nonsensical
        pytest.skip()
    return dict.fromkeys(plot_context.ensembles(), request.param)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_histogram(plot_context: PlotContext, ensemble_to_data_map):
    figure = Figure()
    HistogramPlot().plot(
        figure,
        plot_context,
        ensemble_to_data_map,
        pd.DataFrame(),
        {},
    )
    return figure


def test_histogram_plot_for_constant_distribution(monkeypatch):
    # test that the histogram plot is called with the correct min and max values
    # when all the parameter values are the same
    context = Mock(spec=PlotContext)
    context.ensembles.return_value = [
        EnsembleObject(
            "ensemble_1", "id", False, "experiment_1", started_at="2012-12-10T00:00:00"
        )
    ]
    context.ensembles_color_indexes.return_value = [1]
    title = "Histogram with same values"
    context.plotConfig.return_value = PlotConfig(title=title)
    value = 0
    data_map = dict.fromkeys(context.ensembles(), pd.DataFrame([10 * [value]]))
    min_value = value - 0.1
    max_value = value + 0.1
    figure = Figure()
    mock_plot_histogram = Mock()
    monkeypatch.setattr(
        ert.gui.tools.plot.plottery.plots.histogram,
        "_plotHistogram",
        mock_plot_histogram,
    )
    HistogramPlot().plot(
        figure,
        context,
        data_map,
        pd.DataFrame(),
        {},
    )
    mock_plot_histogram.assert_called_once_with(
        ANY,
        ANY,
        ANY,
        ANY,
        min_value,
        max_value,
    )
