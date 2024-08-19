from unittest.mock import Mock

import pandas as pd
import pytest
from matplotlib.figure import Figure

from ert.gui.plottery import PlotConfig, PlotContext
from ert.gui.plottery.plots.histogram import HistogramPlot
from ert.gui.tools.plot.plot_api import EnsembleObject


@pytest.fixture(
    params=[
        pytest.param(
            (True, [EnsembleObject("ensemble_1", "id", False, "experiment_1")]),
            id="log_scale",
        ),
        pytest.param(
            (False, [EnsembleObject("ensemble_1", "id", False, "experiment_1")]),
            id="no_log_scale",
        ),
        pytest.param((False, []), id="no_ensembles"),
    ]
)
def plot_context(request):
    context = Mock(spec=PlotContext)
    context.log_scale = request.param[0]
    context.ensembles.return_value = request.param[1]
    title = (
        "log_scale"
        if context.log_scale
        else "" + f"num_ensembles={len(request.param[1])}"
    )
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
    if (
        not request.param.empty
        and request.param[0].dtype == "object"
        and plot_context.log_scale
    ):
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
