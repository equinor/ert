import pandas as pd
import pytest
from tests.everest.unit_tests.plots.utils import create_everest_figure, move_cursor

from ert.gui.plotting.everest_plots import EverestControlsPlot
from ert.gui.plotting.utils.plot_context import PlotType


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


@pytest.mark.parametrize(
    ("selected_controls", "expected_line_amount"),
    [
        (["ctrl_a", "ctrl_b"], 2),
        (["ctrl_a"], 1),
    ],
)
def test_that_plot_pr_batch_creates_one_line_per_selected_control(
    generic_plot_context,
    everest_ensemble,
    controls_data,
    selected_controls,
    expected_line_amount,
):
    plot = EverestControlsPlot()
    plot.set_selected_controls(selected_controls)
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    axes = figure.get_axes()[0]
    lines = axes.get_lines()
    assert len(lines) == expected_line_amount


def test_that_by_batch_plot_uses_batch_id_for_x_and_control_value_for_y(
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


@pytest.mark.parametrize(
    ("expected_collections", "expected_lines", "by_batch"),
    [
        (2, 0, False),
        (0, 2, True),
    ],
)
def test_that_plot_creates_lines_in_by_batch_and_scatter_collections_in_by_control(
    generic_plot_context,
    everest_ensemble,
    controls_data,
    expected_collections,
    expected_lines,
    by_batch,
):
    generic_plot_context.by_batch = by_batch
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a", "ctrl_b"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    axes = figure.get_axes()[0]
    collections = axes.collections
    lines = axes.get_lines()
    assert len(collections) == expected_collections
    assert len(lines) == expected_lines


@pytest.mark.parametrize(
    ("by_batch", "plottype"), [(True, PlotType.LINE), (False, PlotType.SCATTER)]
)
def test_that_plot_with_empty_data_have_correct_plot_type_and_creates_helper_text(
    generic_plot_context, everest_ensemble, by_batch, plottype
):
    generic_plot_context.by_batch = by_batch
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])
    figure = create_everest_figure(
        plot, pd.DataFrame(), generic_plot_context, everest_ensemble
    )
    assert generic_plot_context.plot_type == plottype
    assert figure.get_axes()[0].texts[0].get_text() == "No data"


@pytest.mark.parametrize(
    (
        "collections_x_value",
        "collections_y_value",
        "lines_x_value",
        "lines_y_value",
        "by_batch",
    ),
    [
        ([0.0], [1.0], None, None, False),
        (None, None, [0, 1], [1.0, 1.5], True),
    ],
)
def test_that_plot_uses_expected_x_and_y_values_in_by_batch_and_by_control_modes(
    generic_plot_context,
    everest_ensemble,
    controls_data,
    collections_x_value,
    collections_y_value,
    lines_x_value,
    lines_y_value,
    by_batch,
):
    generic_plot_context.by_batch = by_batch
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    axes = figure.get_axes()[0]
    if by_batch:
        lines = axes.get_lines()
        assert len(lines) == 1
        line = lines[0]
        assert list(line.get_xdata()) == lines_x_value
        assert list(line.get_ydata()) == lines_y_value
    else:
        offsets = axes.collections[0].get_offsets()
        assert len(offsets) == 1
        assert list(offsets[:, 0]) == collections_x_value
        assert list(offsets[:, 1]) == collections_y_value


@pytest.mark.parametrize(
    ("is_visible", "number_of_legend_items", "increased_data"),
    [
        (True, 2, False),
        (False, 0, True),
    ],
)
def test_that_legend_is_hidden_when_number_of_controls_exceeds_legend_threshold(
    generic_plot_context,
    everest_ensemble,
    controls_data,
    many_controls,
    is_visible,
    number_of_legend_items,
    increased_data,
    palette_size,
):
    plot = EverestControlsPlot()
    plot.set_selected_controls(
        [f"ctrl_{i}" for i in range(palette_size * 4)]
        if increased_data
        else ["ctrl_a", "ctrl_b"]
    )
    figure = create_everest_figure(
        plot,
        many_controls if increased_data else controls_data,
        generic_plot_context,
        everest_ensemble,
    )

    legend = figure.get_axes()[0].get_legend()
    assert (legend is not None) == is_visible
    if is_visible:
        legend_texts = [text.get_text() for text in legend.get_texts()]
        assert len(legend_texts) == number_of_legend_items
        assert legend_texts == ["ctrl_a", "ctrl_b"]


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
    assert len(lines) == palette_size * 4
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


@pytest.mark.parametrize(
    ("by_batch", "expected_plot_type"),
    [
        (True, PlotType.LINE),
        (False, PlotType.SCATTER),
    ],
)
def test_that_plot_type_matches_by_batch_setting(
    by_batch, expected_plot_type, generic_plot_context, everest_ensemble, controls_data
):
    generic_plot_context.by_batch = by_batch
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])
    create_everest_figure(plot, controls_data, generic_plot_context, everest_ensemble)

    assert generic_plot_context.plot_type == expected_plot_type


def test_that_date_support_is_deactivated(
    generic_plot_context, everest_ensemble, controls_data
):
    plot = EverestControlsPlot()
    plot.set_selected_controls(["ctrl_a"])
    create_everest_figure(plot, controls_data, generic_plot_context, everest_ensemble)

    assert not generic_plot_context.is_date_support_active()


def test_that_hover_annotation_displays_control_name_for_by_batch_plot(
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
