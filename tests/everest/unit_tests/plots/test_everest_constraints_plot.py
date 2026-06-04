import pandas as pd
import pytest
from tests.everest.unit_tests.plots.utils import create_everest_figure

from ert.gui.tools.plot.plottery.plots import EverestConstraintsPlot


@pytest.fixture
def constraints_data():
    return pd.DataFrame(
        {
            "batch_id": [0, 1],
            "realization": [0, 0],
            "constraint_value": [0.1, 0.2],
            "lower_bound": [0.05, 0.05],
            "upper_bound": [0.25, 0.25],
        }
    )


def test_that_bounds_and_realization_are_plotted(
    everest_ensemble, generic_plot_context, constraints_data
):
    figure = create_everest_figure(
        EverestConstraintsPlot(),
        constraints_data,
        generic_plot_context,
        everest_ensemble,
    )

    axes = figure.axes[0]
    lines = axes.get_lines()
    assert len(lines) == 3  # Realization line + lower bound + upper bound
    line1, line2, line3 = lines
    assert list(line1.get_xdata()) == [0, 1]
    assert list(line1.get_ydata()) == [0.1, 0.2]  # Realization value
    assert list(line2.get_xdata()) == [0, 1]
    assert list(line2.get_ydata()) == [0.05, 0.05]  # Lower bound
    assert list(line3.get_xdata()) == [0, 1]
    assert list(line3.get_ydata()) == [0.25, 0.25]  # Upper bound


def test_that_bounds_have_correct_style(
    everest_ensemble, generic_plot_context, constraints_data
):
    figure = create_everest_figure(
        EverestConstraintsPlot(),
        constraints_data,
        generic_plot_context,
        everest_ensemble,
    )

    axes = figure.get_axes()[0]
    lines = axes.get_lines()
    bound_lines = lines[1:]
    assert len(bound_lines) > 0
    for line in bound_lines:
        assert line.get_linestyle() == "--"

    spans = axes.patches
    assert len(spans) == 2
