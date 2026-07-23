import pandas as pd
import pytest

from ert.gui.plotting.everest_plots import EverestConstraintsPlot
from ert.gui.plotting.utils.plot_context import PlotType
from tests.everest.unit_tests.plots.utils import create_everest_figure, move_cursor


def generate_constraints_data(num_realizations: int):
    return pd.DataFrame(
        {
            "batch_id": [0, 1] * num_realizations,
            "realization": list(range(num_realizations)) * 2,
            "constraint_value": [0.1, 0.2] * num_realizations,
            "lower_bound": [0.05, 0.05] * num_realizations,
            "upper_bound": [0.25, 0.25] * num_realizations,
        }
    )


def num_of_lines_to_be_plotted_incl_bound_lines(df: pd.DataFrame):
    num_realizations = len(df["realization"].unique())
    return num_realizations + 2


def test_that_plot_type_is_set_to_line(generic_plot_context, everest_ensemble):
    create_everest_figure(
        EverestConstraintsPlot(),
        generate_constraints_data(1),
        generic_plot_context,
        everest_ensemble,
    )
    assert generic_plot_context.plot_type == PlotType.LINE


@pytest.mark.parametrize(
    ("realization_data", "expected_num_of_colors"),
    [
        (
            generate_constraints_data(1),
            num_of_lines_to_be_plotted_incl_bound_lines(generate_constraints_data(1))
            - 1,
        ),  # One color for each realization line, one for bounds
        (generate_constraints_data(6), 2),
    ],
)
def test_that_unique_colors_only_if_num_of_realizations_is_at_or_below_legend_threshold(
    realization_data, expected_num_of_colors, generic_plot_context, everest_ensemble
):
    figure = create_everest_figure(
        EverestConstraintsPlot(),
        realization_data,
        generic_plot_context,
        everest_ensemble,
    )
    colors = [line.get_color() for line in figure.get_axes()[0].get_lines()]
    assert len(set(colors)) == expected_num_of_colors


@pytest.mark.parametrize(
    ("realization_data", "expected_legend_length"),
    [
        (
            generate_constraints_data(5),
            num_of_lines_to_be_plotted_incl_bound_lines(generate_constraints_data(5))
            - 1,
        ),  # One legend item for each realization line, one for bounds
        (generate_constraints_data(6), 1),  # Only bounds in legend
    ],
)
def test_that_legend_items_are_conditionally_shown(
    generic_plot_context, everest_ensemble, realization_data, expected_legend_length
):
    figure = create_everest_figure(
        EverestConstraintsPlot(),
        realization_data,
        generic_plot_context,
        everest_ensemble,
    )
    legend = figure.get_axes()[0].get_legend()
    legend_length = len(legend.get_texts()) if legend else None
    assert legend_length == expected_legend_length


def test_that_plot_type_is_set_to_line_and_line_style_is_correct(
    generic_plot_context, everest_ensemble
):
    figure = create_everest_figure(
        EverestConstraintsPlot(),
        generate_constraints_data(1),
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
    everest_ensemble, generic_plot_context
):
    figure = create_everest_figure(
        EverestConstraintsPlot(),
        generate_constraints_data(1),
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


def test_that_bounds_have_correct_style(everest_ensemble, generic_plot_context):
    figure = create_everest_figure(
        EverestConstraintsPlot(),
        generate_constraints_data(1),
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


def test_that_tooltip_shows_on_hover(everest_ensemble, generic_plot_context):
    figure = create_everest_figure(
        EverestConstraintsPlot(),
        generate_constraints_data(1),
        generic_plot_context,
        everest_ensemble,
    )

    axes = figure.get_axes()[0]

    move_cursor(
        axes,
        x=0.0,
        y=0.1,
    )

    annotations = axes.texts
    assert len(annotations) == 1

    hover_annotation = annotations[0]

    assert hover_annotation.get_text() == "Realization 0"
    assert hover_annotation.get_visible()

    move_cursor(
        axes,
        x=0.0,
        y=1.0,
    )

    assert not hover_annotation.get_visible()


@pytest.mark.mpl_image_compare(tolerance=10.0, style="default")
def test_that_constraints_plot_matches_baseline(everest_ensemble, generic_plot_context):
    return create_everest_figure(
        EverestConstraintsPlot(),
        generate_constraints_data(1),
        generic_plot_context,
        everest_ensemble,
    )
