import pandas as pd
import pytest
from tests.everest.unit_tests.plots.utils import create_everest_figure, move_cursor

from ert.gui.tools.plot.plottery.plot_context import PlotType
from ert.gui.tools.plot.plottery.plots import EverestObjectiveFunctionPlot
from ert.gui.utils import LEGEND_THRESHOLD


def realization_data():
    return pd.DataFrame(
        {
            "batch_id": [0] * LEGEND_THRESHOLD + [1] * LEGEND_THRESHOLD,
            "realization": list(range(LEGEND_THRESHOLD)) * 2,
            "objective_function_value": [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
        }
    )


def lots_of_realization_data():
    n_realizations = LEGEND_THRESHOLD + 1
    return pd.DataFrame(
        {
            "batch_id": [0] * n_realizations + [1] * n_realizations,
            "realization": list(range(n_realizations)) * 2,
            "objective_function_value": [float(i) for i in range(n_realizations)]
            + [float(i) + 0.5 for i in range(n_realizations)],
        }
    )


def test_that_plot_type_is_set_to_line_and_style_is_correct(
    generic_plot_context, everest_ensemble
):
    figure = create_everest_figure(
        EverestObjectiveFunctionPlot(),
        realization_data(),
        generic_plot_context,
        everest_ensemble,
    )

    assert generic_plot_context.plot_type == PlotType.LINE
    lines = figure.get_axes()[0].get_lines()
    assert len(lines) > 0
    for line in lines:
        assert line.get_linestyle() == "-"
        assert line.get_marker() == "o"


@pytest.mark.parametrize(
    ("realization_data", "expected_num_lines"),
    [
        (realization_data(), LEGEND_THRESHOLD),
        (lots_of_realization_data(), LEGEND_THRESHOLD + 1),
    ],
)
def test_that_all_realizations_are_plotted(
    generic_plot_context, everest_ensemble, realization_data, expected_num_lines
):
    figure = create_everest_figure(
        EverestObjectiveFunctionPlot(),
        realization_data,
        generic_plot_context,
        everest_ensemble,
    )
    assert len(figure.get_axes()[0].get_lines()) == expected_num_lines


@pytest.mark.parametrize(
    ("realization_data", "expected_num_of_colors"),
    [
        (realization_data(), LEGEND_THRESHOLD),
        (lots_of_realization_data(), 1),
    ],
)
def test_that_color_is_correctly_set_based_on_amount_of_realizations(
    realization_data, expected_num_of_colors, generic_plot_context, everest_ensemble
):
    figure = create_everest_figure(
        EverestObjectiveFunctionPlot(),
        realization_data,
        generic_plot_context,
        everest_ensemble,
    )
    colors = [line.get_color() for line in figure.get_axes()[0].get_lines()]
    assert len(set(colors)) == expected_num_of_colors


@pytest.mark.parametrize(
    ("realization_data", "expected_legend_length"),
    [
        (realization_data(), LEGEND_THRESHOLD),
        (lots_of_realization_data(), None),
    ],
)
def test_that_legend_is_conditionally_shown(
    generic_plot_context, everest_ensemble, realization_data, expected_legend_length
):
    figure = create_everest_figure(
        EverestObjectiveFunctionPlot(),
        realization_data,
        generic_plot_context,
        everest_ensemble,
    )
    legend = figure.get_axes()[0].get_legend()
    legend_length = len(legend.get_texts()) if legend else None
    assert legend_length == expected_legend_length


def test_that_on_hover_labels_are_correct(generic_plot_context, everest_ensemble):
    figure = create_everest_figure(
        EverestObjectiveFunctionPlot(),
        realization_data(),
        generic_plot_context,
        everest_ensemble,
    )
    axes = figure.get_axes()[0]

    move_cursor(
        axes=axes,
        x=0.5,
        y=15,
    )

    annotations = [text for text in axes.texts if "Realization 0" in text.get_text()]
    hover_annotation = annotations[0]
    assert len(annotations) == 1
    assert hover_annotation.get_text() == "Realization 0"
    assert hover_annotation.get_visible()

    move_cursor(
        axes=axes,
        x=0,
        y=0,
    )

    assert not hover_annotation.get_visible()
