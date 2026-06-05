import pandas as pd
import pytest
from tests.everest.unit_tests.plots.utils import create_everest_figure, move_cursor

from ert.gui.tools.plot.plottery.plots import EverestBatchObjectiveFunctionPlot


@pytest.fixture
def batch_objective_data():
    return pd.DataFrame(
        {
            "batch_id": [0, 1, 2, 3],
            "objective_function_value": [8.0, 9.0, 10.0, 7.0],
            "is_improvement": [True, True, False, True],
            "improvement_value": [float("-inf"), 2.0, float("nan"), 2.0],
            "constraint_violation_type": [
                None,
                None,
                "bound constraint violation",
                "non-improvement",
            ],
            "constraint_violation_value": [
                None,
                None,
                0.3,
                None,
            ],
        }
    )


def test_that_date_support_is_disabled(generic_plot_context):
    generic_plot_context.deactivate_date_support()
    assert not generic_plot_context.is_date_support_active()


def test_that_empty_data_returns_early_with_helper_text(
    generic_plot_context, everest_ensemble
):
    figure = create_everest_figure(
        EverestBatchObjectiveFunctionPlot(),
        pd.DataFrame(),
        generic_plot_context,
        everest_ensemble,
    )
    assert len(figure.get_axes()[0].get_lines()) == 0
    assert figure.get_axes()[0].texts[0].get_text() == "No data"


def test_that_plot_adds_accepted_line_and_scatter_for_accepted_and_rejected_batches(
    everest_ensemble, batch_objective_data, generic_plot_context
):
    figure = create_everest_figure(
        EverestBatchObjectiveFunctionPlot(),
        batch_objective_data,
        generic_plot_context,
        everest_ensemble,
    )

    axes = figure.get_axes()[0]
    lines = axes.get_lines()
    assert len(lines) == 1
    line = lines[0]
    assert line.get_color() == generic_plot_context.plotConfig().current_color()
    assert line.get_linestyle() == "-"
    assert line.get_marker() == "o"

    collections = axes.collections
    assert len(collections) == 2


def test_that_hovering_over_points_shows_correct_tooltip(
    everest_ensemble, batch_objective_data, generic_plot_context
):
    figure = create_everest_figure(
        EverestBatchObjectiveFunctionPlot(),
        batch_objective_data,
        generic_plot_context,
        everest_ensemble,
    )

    axes = figure.get_axes()[0]

    move_cursor(axes=axes, x=0, y=8.0)

    annotations = list(axes.texts)
    assert len(annotations) == 2
    rejected_batches = annotations[0]
    accepted_batches = annotations[1]

    accepted_batches_text = accepted_batches.get_text()
    assert "Batch 0" in accepted_batches_text
    assert accepted_batches.get_visible()
    assert not rejected_batches.get_visible()

    move_cursor(axes=axes, x=2, y=10.0)
    assert rejected_batches.get_visible()
    assert "Batch 2" in rejected_batches.get_text()
    assert not accepted_batches.get_visible()
