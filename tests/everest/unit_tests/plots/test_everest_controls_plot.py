import pandas as pd
import pytest
from tests.everest.unit_tests.plots.utils import create_everest_figure, move_cursor

from ert.gui.tools.plot.plottery.plot_context import PlotType
from ert.gui.tools.plot.plottery.plots import EverestControlsPlot


@pytest.fixture
def controls_data():
    return pd.DataFrame(
        {
            "batch_id": [0, 0, 1, 1],
            "realization": [0, 0, 0, 0],
            "control_name": ["ctrl_a", "ctrl_b", "ctrl_a", "ctrl_b"],
            "control_value": [1.0, 2.0, 1.5, 2.5],
        }
    )


@pytest.fixture
def many_controls(palette_size):
    names = [f"ctrl_{i}" for i in range(palette_size * 4)]
    return pd.DataFrame(
        {
            "batch_id": [0] * palette_size * 4 + [1] * palette_size * 4,
            "realization": [0] * 2 * palette_size * 4,
            "control_name": names * 2,
            "control_value": [float(i) for i in range(palette_size * 4)]
            + [float(i) + 0.5 for i in range(palette_size * 4)],
        }
    )


def test_that_plot_pr_batch_creates_one_line_per_selected_control(
    generic_plot_context, everest_ensemble, controls_data
):
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    axes = figure.get_axes()[0]
    lines = axes.get_lines()
    assert len(lines) == 2


def test_that_plot_pr_batch_creates_one_for_single_control(
    generic_plot_context, everest_ensemble, controls_data
):
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    axes = figure.get_axes()[0]
    lines = axes.get_lines()
    assert len(lines) == 1
    line = lines[0]
    assert list(line.get_xdata()) == [0, 1]
    assert list(line.get_ydata()) == [1.0, 1.5]


def test_that_no_collections_are_created_when_by_batch_is_true(
    generic_plot_context, everest_ensemble, controls_data
):
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    axes = figure.get_axes()[0]
    collections = axes.collections
    assert len(collections) == 0


def test_that_no_lines_are_created_when_by_batch_is_false(
    generic_plot_context, everest_ensemble, controls_data
):
    generic_plot_context.by_batch = False
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    axes = figure.get_axes()[0]
    lines = axes.get_lines()
    assert len(lines) == 0


def test_that_plot_pr_control_creates_one_scatter_collection_per_batch(
    generic_plot_context, everest_ensemble, controls_data
):
    generic_plot_context.by_batch = False
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    axes = figure.get_axes()[0]
    collections = axes.collections
    assert len(collections) == 2


def test_that_plot_pr_batch_with_empty_data_creates_no_axes(
    generic_plot_context, everest_ensemble
):
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])
    figure = create_everest_figure(
        plot, pd.DataFrame(), generic_plot_context, everest_ensemble
    )

    assert len(figure.get_axes()) == 0


def test_that_plot_pr_batch_uses_correct_x_and_y_values(
    generic_plot_context, everest_ensemble, controls_data
):
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    axes = figure.get_axes()[0]
    line = axes.get_lines()[0]
    # x is batch_id, y is control_value for ctrl_a
    assert list(line.get_xdata()) == [0, 1]
    assert list(line.get_ydata()) == [1.0, 1.5]


def test_that_plot_pr_control_uses_correct_x_and_y_values(
    generic_plot_context, everest_ensemble, controls_data
):
    generic_plot_context.by_batch = False
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    axes = figure.get_axes()[0]
    offsets_0 = axes.collections[0].get_offsets()
    assert len(offsets_0) == 2
    assert list(offsets_0[:, 0]) == [0.0, 1.0]  # x positions
    assert list(offsets_0[:, 1]) == [1.0, 2.0]  # y values

    offsets_1 = axes.collections[1].get_offsets()
    assert len(offsets_1) == 2
    assert list(offsets_1[:, 0]) == [0.0, 1.0]
    assert list(offsets_1[:, 1]) == [1.5, 2.5]


def test_that_plot_contains_legend_items_when_number_of_entries_is_below_threshold(
    generic_plot_context, everest_ensemble, controls_data
):
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    legend = figure.get_axes()[0].get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert legend_texts == ["ctrl_a", "ctrl_b"]


def test_that_plot_does_not_contain_legend_when_number_of_entries_is_above_threshold(
    generic_plot_context, everest_ensemble, many_controls, palette_size
):
    plot = EverestControlsPlot()
    plot.set_selected_controls([f"ctrl_{i}" for i in range(palette_size * 4)])
    figure = create_everest_figure(
        plot, many_controls, generic_plot_context, everest_ensemble
    )

    assert figure.get_axes()[0].get_legend() is None


def test_that_plot_rename_batch_0_in_legend_for_by_controls_plot(
    generic_plot_context, everest_ensemble, controls_data
):
    generic_plot_context.by_batch = False
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    legend = figure.get_axes()[0].get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "initial batch" in legend_texts
    assert "batch_1" in legend_texts


def test_that_line_style_matches_when_number_of_controls_exceeds_color_palette_length(
    generic_plot_context, everest_ensemble, many_controls, palette_size
):
    plot = EverestControlsPlot()
    plot.set_selected_controls([f"ctrl_{i}" for i in range(palette_size * 4)])
    figure = create_everest_figure(
        plot, many_controls, generic_plot_context, everest_ensemble
    )

    axes = figure.get_axes()[0]
    lines = axes.get_lines()
    for i, line in enumerate(lines):
        if i < palette_size:
            assert line.get_linestyle() == "-"
            assert line.get_marker() == "o"
        elif i < palette_size * 2:
            assert line.get_linestyle() == "--"
            assert line.get_marker() == "o"
        elif i < palette_size * 3:
            assert line.get_linestyle() == ":"
            assert line.get_marker() == "o"
        else:
            assert line.get_linestyle() == "-."
            assert line.get_marker() == "o"


def test_that_plot_type_matches_line_when_by_batch_is_true(
    generic_plot_context, everest_ensemble, controls_data
):
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])
    create_everest_figure(plot, controls_data, generic_plot_context, everest_ensemble)

    assert generic_plot_context.plot_type == PlotType.LINE


def test_that_plot_type_matches_scatter_when_by_batch_is_false(
    generic_plot_context, everest_ensemble, controls_data
):
    generic_plot_context.by_batch = False
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])
    create_everest_figure(plot, controls_data, generic_plot_context, everest_ensemble)

    assert generic_plot_context.plot_type == PlotType.SCATTER


def test_that_date_support_is_deactivated(
    generic_plot_context, everest_ensemble, controls_data
):
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])
    create_everest_figure(plot, controls_data, generic_plot_context, everest_ensemble)

    assert not generic_plot_context.is_date_support_active()


def test_that_hover_labels_are_set_correctly_for_by_batch_plot(
    generic_plot_context, everest_ensemble, controls_data
):
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    axes = figure.get_axes()[0]

    move_cursor(
        axes=axes,
        x=0.5,
        y=1.25,
    )

    annotations = [text for text in axes.texts if "ctrl_a" in text.get_text()]
    hover_annotation = annotations[1]  # Will be the second annotation
    # First annotation is the right hand side label for the line
    assert len(annotations) == 2
    assert hover_annotation.get_text() == "ctrl_a"
    assert hover_annotation.get_visible()

    move_cursor(
        axes=axes,
        x=0,
        y=0,
    )

    assert not hover_annotation.get_visible()
