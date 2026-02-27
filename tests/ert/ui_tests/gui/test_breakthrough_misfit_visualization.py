import pytest

from ert.gui.tools.plot.plot_window import MISFITS
from tests.ert.ui_tests.gui.test_breakthrough_visualization import (
    create_breakthrough_figure,
)

plot_figure = create_breakthrough_figure(MISFITS)


@pytest.mark.mpl_image_compare(tolerance=10.0)
@pytest.mark.skip_mac_ci
@pytest.mark.snapshot_test
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_breakthrough_misfit_visualization_matches_snapshot(plot_figure):
    return plot_figure
