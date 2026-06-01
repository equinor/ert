import pytest

from ert.gui.tools.plot.plottery.plot_config import PlotConfig
from ert.gui.tools.plot.plottery.plot_context import PlotContext


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
