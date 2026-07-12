import pytest

from ert.gui.plotting.plot_api import EnsembleObject
from ert.gui.plotting.utils import PlotConfig, PlotContext


@pytest.fixture
def everest_ensemble():
    return EnsembleObject(
        "batch_0", "id", False, "experiment_1", started_at="2012-12-10T00:00:00"
    )


@pytest.fixture
def generic_plot_context():
    plot_config = PlotConfig()

    return PlotContext(
        plot_config,
        ensembles=[],
        ensembles_color_indexes=[],
        key="test",
    )


@pytest.fixture
def palette_size(generic_plot_context):
    return generic_plot_context.plotConfig().get_number_of_colors()
