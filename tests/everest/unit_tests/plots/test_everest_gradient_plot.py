import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.container import BarContainer
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
    fig, plot = plt.figure(), EverestGradientsPlot()
    plot.plot(fig, generic_plot_context, {}, pd.DataFrame(), {}, None, None)

    assert (
        fig.axes[0].texts[0].get_text()
        == "Select control(s) from the right side panel to view gradients"
    )


def test_gradient_plot_with_no_data_generates_helper_text(generic_plot_context):
    fig, plot = plt.figure(), EverestGradientsPlot()
    plot.set_selected_controls(["control_1"])
    plot.plot(fig, generic_plot_context, {}, pd.DataFrame(), {}, None, None)

    assert fig.axes[0].texts[0].get_text() == "No data"


def test_gradient_plot_with_control_selected(generic_plot_args):
    fig, plot = plt.figure(), EverestGradientsPlot()
    plot.set_selected_controls(["control_1"])
    plot.plot(fig, *generic_plot_args)

    bars = [b for b in fig.get_axes()[0].containers if isinstance(b, BarContainer)]
    assert len(bars) == 1

    children = bars[0].patches
    assert len(children) == 2

    assert children[0].get_height() == pytest.approx(0.5)
    assert children[1].get_height() == pytest.approx(0.7)


def test_gradient_plot_legends_appear(generic_plot_args):
    fig, plot = plt.figure(), EverestGradientsPlot()
    plot.set_selected_controls(["control_1", "control_2"])
    plot.plot(fig, *generic_plot_args)

    legend = fig.get_axes()[0].get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "control_1" in legend_texts
    assert "control_2" in legend_texts


def test_gradient_plot_hatches(generic_plot_context, ensemble):
    fig, plot = plt.figure(), EverestGradientsPlot()

    data = pd.DataFrame(
        {
            "batch_id": list(range(30)),
            "realization": list(range(30)),
            "control_name": [f"control_{i}" for i in range(30)],
            "control_value": [i * 0.1 for i in range(30)],
            "Response": list(range(30)),
        }
    )
    n_colors = generic_plot_context.plotConfig().get_number_of_colors()
    plot.set_selected_controls([f"control_{i}" for i in range(30)])
    plot.plot(
        fig, generic_plot_context, {ensemble: data}, pd.DataFrame(), {}, None, None
    )
    bars = [b for b in fig.get_axes()[0].containers if isinstance(b, BarContainer)]
    hatches = [b.patches[0].get_hatch() for b in bars]
    for i, patch in enumerate(hatches):
        if i < n_colors:
            assert not patch
        elif i < n_colors * 2:
            assert patch == "//"
        elif i < n_colors * 3:
            assert patch == ".."
        else:
            assert patch == "-"


def test_gradient_plot_on_hover_functionality(generic_plot_args):
    fig, plot = plt.figure(), EverestGradientsPlot()
    plot.set_selected_controls(["control_1"])
    plot.plot(fig, *generic_plot_args)

    ax = fig.get_axes()[0]
    bars = [b for b in ax.containers if isinstance(b, BarContainer)]
    bar = bars[0].patches[0]

    move_cursor(
        ax=ax,
        x=bar.get_x() + bar.get_width() / 2,
        y=bar.get_height() / 2,
    )

    annotation = [text for text in ax.texts if "batch 0\n" in text.get_text()]
    assert len(annotation) == 1
    assert annotation[0].get_text() == "batch 0\ncontrol_1\nValue: 0.5"
    assert annotation[0].get_visible()
