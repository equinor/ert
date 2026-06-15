import pandas as pd
import pytest
from tests.everest.unit_tests.plots.utils import create_everest_figure

from ert.gui.plotting.everest_plots import EverestConstraintsPlot
from ert.gui.plotting.utils.plot_context import PlotType


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


def test_that_plot_type_is_set_to_line_and_line_style_is_correct(
    generic_plot_context, everest_ensemble, constraints_data
):
    figure = create_everest_figure(
        EverestConstraintsPlot(),
        constraints_data,
        generic_plot_context,
        everest_ensemble,
    )

    assert generic_plot_context.plot_type == PlotType.LINE
    axes = figure.get_axes()[0]
    lines = axes.get_lines()
    assert len(lines) == 3  # constraint_value + lower_bound  + upper_bound
    value_line = lines[0]
    assert value_line.get_linestyle() == "-"
    assert value_line.get_marker() == "o"


def test_that_bounds_and_realization_are_plotted(
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
    assert len(lines) == 3  # Realization line + lower bound + upper bound
    line1, line2, line3 = lines
    assert list(line1.get_xdata()) == [0, 1]
    assert list(line1.get_ydata()) == [0.1, 0.2]  # Realization value
    assert list(line2.get_xdata()) == [0, 1]
    assert list(line2.get_ydata()) == [0.05, 0.05]  # Lower bound
    assert list(line3.get_xdata()) == [0, 1]
    assert list(line3.get_ydata()) == [0.25, 0.25]  # Upper bound


def test_that_empty_data_returns_early_with_helper_text(
    generic_plot_context, everest_ensemble
):
    figure = create_everest_figure(
        EverestConstraintsPlot(),
        pd.DataFrame(),
        generic_plot_context,
        everest_ensemble,
    )
    assert len(figure.get_axes()[0].get_lines()) == 0
    assert figure.get_axes()[0].texts[0].get_text() == "No data"


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
        assert line.get_marker() == "None"

    spans = axes.patches
    assert len(spans) == 2
