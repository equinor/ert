from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from ert.gui.tools.plot.plot_api import EnsembleObject
from ert.gui.tools.plot.plottery import PlotConfig, PlotContext
from ert.gui.tools.plot.plottery.plots import StdDevPlot


@pytest.fixture()
def plot_context(request):
    context = Mock(spec=PlotContext)
    context.ensembles.return_value = [
        EnsembleObject(
            "ensemble_1", "id", False, "experiment_1", started_at="2012-12-10T00:00:00"
        )
    ]
    context.history_data = None
    context.layer = 0
    context.plotConfig.return_value = PlotConfig(title="StdDev Plot")
    return context


def test_stddev_plot_shows_boxplot(plot_context: PlotContext):
    rng = np.random.default_rng()
    figure = Figure()
    std_dev_data = rng.random((5, 5))
    StdDevPlot().plot(
        figure,
        plot_context,
        {},
        {},
        {"ensemble_1": std_dev_data},
    )
    ax = figure.axes
    assert ax[0].get_title() == "experiment_1 : ensemble_1 layer=0"
    assert ax[1].get_ylabel() == "Standard Deviation"
    annotation = [
        child for child in ax[1].get_children() if isinstance(child, plt.Annotation)
    ]
    assert len(annotation) == 1
    min_value = np.min(std_dev_data)
    mean_value = np.mean(std_dev_data)
    max_value = np.max(std_dev_data)
    assert (
        annotation[0].get_text()
        == f"Min: {min_value:.2f}\nMean: {mean_value:.2f}\nMax: {max_value:.2f}"
    )
