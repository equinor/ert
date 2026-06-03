import pandas as pd
import pytest
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from tests.everest.unit_tests.plots.utils import move_cursor

from ert.gui.tools.plot.plot_api import EnsembleObject
from ert.gui.tools.plot.plottery.plots.everest_gradients_plot import (
    EverestGradientsPlot,
)


@pytest.fixture
def ensemble():
    return EnsembleObject(
        "ensemble_1", "id", False, "experiment_1", started_at="2012-12-10T00:00:00"
    )


@pytest.fixture
def controls_data():
    return pd.DataFrame(
        {
            "batch_id": [0, 0, 1, 1],
            "realization": [0, 0, 0, 0],
            "control_name": ["control_1", "control_2", "control_1", "control_2"],
            "control_value": [0.1, 0.2, 0.3, 0.4],
            "Response": [0.5, 0.6, 0.7, 0.8],
        }
    )


@pytest.fixture
def generic_plot_args(generic_plot_context, ensemble, controls_data):
    return (
        generic_plot_context,
        {ensemble: controls_data},
        pd.DataFrame(),
        {},
        None,
        None,
    )


def test_gradient_plot_with_zero_controls_selected_generates_helper_text(
    generic_plot_context,
):
    figure, plot = Figure(), EverestGradientsPlot()
    plot.plot(figure, generic_plot_context, {}, pd.DataFrame(), {}, None, None)

    assert (
        figure.axes[0].texts[0].get_text()
        == "Select control(s) from the right side panel to view gradients"
    )


def test_gradient_plot_with_no_data_generates_helper_text(generic_plot_context):
    figure, plot = Figure(), EverestGradientsPlot()
    plot.set_selected_controls(["control_1"])
    plot.plot(figure, generic_plot_context, {}, pd.DataFrame(), {}, None, None)

    assert figure.axes[0].texts[0].get_text() == "No data"


def test_gradient_plot_with_control_selected(generic_plot_args):
    figure, plot = Figure(), EverestGradientsPlot()
    plot.set_selected_controls(["control_1"])
    plot.plot(figure, *generic_plot_args)

    bars = [b for b in figure.get_axes()[0].containers if isinstance(b, BarContainer)]
    assert len(bars) == 1

    children = bars[0].patches
    assert len(children) == 2

    assert children[0].get_height() == pytest.approx(0.5)
    assert children[1].get_height() == pytest.approx(0.7)


def test_gradient_plot_legends_appear(generic_plot_args):
    figure, plot = Figure(), EverestGradientsPlot()
    plot.set_selected_controls(["control_1", "control_2"])
    plot.plot(figure, *generic_plot_args)

    legend = figure.get_axes()[0].get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "control_1" in legend_texts
    assert "control_2" in legend_texts


def test_gradient_plot_hatches(generic_plot_context, ensemble, palette_size):
    figure, plot = Figure(), EverestGradientsPlot()

    data = pd.DataFrame(
        {
            "batch_id": list(range(palette_size * 4)),
            "realization": list(range(palette_size * 4)),
            "control_name": [f"control_{i}" for i in range(palette_size * 4)],
            "control_value": [i * 0.1 for i in range(palette_size * 4)],
            "Response": list(range(palette_size * 4)),
        }
    )
    plot.set_selected_controls([f"control_{i}" for i in range(palette_size * 4)])
    plot.plot(
        figure, generic_plot_context, {ensemble: data}, pd.DataFrame(), {}, None, None
    )
    bars = [b for b in figure.get_axes()[0].containers if isinstance(b, BarContainer)]
    hatches = [b.patches[0].get_hatch() for b in bars]
    for i, patch in enumerate(hatches):
        if i < palette_size:
            assert not patch
        elif i < palette_size * 2:
            assert patch == "//"
        elif i < palette_size * 3:
            assert patch == ".."
        else:
            assert patch == "-"


def test_gradient_plot_on_hover_functionality(generic_plot_args):
    figure, plot = Figure(), EverestGradientsPlot()
    plot.set_selected_controls(["control_1"])
    plot.plot(figure, *generic_plot_args)

    axes = figure.get_axes()[0]
    bars = [b for b in axes.containers if isinstance(b, BarContainer)]
    bar = bars[0].patches[0]

    move_cursor(
        axes=axes,
        x=bar.get_x() + bar.get_width() / 2,
        y=bar.get_height() / 2,
    )

    annotations = [text for text in axes.texts if "batch 0\n" in text.get_text()]
    hover_annot = annotations[0]

    assert len(annotations) == 1
    assert hover_annot.get_text() == "batch 0\ncontrol_1\nValue: 0.5"
    assert hover_annot.get_visible()

    move_cursor(
        axes=axes,
        x=-5.0,
        y=-5.0,
    )
    assert not hover_annot.get_visible()
