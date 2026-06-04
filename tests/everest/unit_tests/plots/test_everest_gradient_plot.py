import pandas as pd
import pytest
from matplotlib.container import BarContainer
from tests.everest.unit_tests.plots.utils import create_everest_figure, move_cursor

from ert.gui.tools.plot.plottery.plots.everest_gradients_plot import (
    EverestGradientsPlot,
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


@pytest.mark.parametrize(
    ("selected_controls", "expected_text"),
    [
        ([], "Select control(s) from the right side panel to view gradients"),
        (["control_1"], "No data"),
    ],
)
def test_gradient_plot_with_no_controls_or_data_shows_helper_text(
    generic_plot_context, everest_ensemble, selected_controls, expected_text
):
    plot = EverestGradientsPlot()
    plot.set_selected_controls(selected_controls)
    figure = create_everest_figure(
        plot, pd.DataFrame(), generic_plot_context, everest_ensemble
    )
    assert figure.axes[0].texts[0].get_text() == expected_text


def test_gradient_plot_with_control_selected(
    controls_data, generic_plot_context, everest_ensemble
):
    plot = EverestGradientsPlot()
    plot.set_selected_controls(["control_1"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    bars = [b for b in figure.get_axes()[0].containers if isinstance(b, BarContainer)]
    assert len(bars) == 1

    children = bars[0].patches
    assert len(children) == 2

    assert children[0].get_height() == pytest.approx(0.5)
    assert children[1].get_height() == pytest.approx(0.7)


def test_gradient_plot_legends_appear(
    controls_data, generic_plot_context, everest_ensemble
):
    plot = EverestGradientsPlot()
    plot.set_selected_controls(["control_1", "control_2"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

    legend = figure.get_axes()[0].get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "control_1" in legend_texts
    assert "control_2" in legend_texts


def test_gradient_plot_hatches(generic_plot_context, everest_ensemble, palette_size):
    data = pd.DataFrame(
        {
            "batch_id": list(range(palette_size * 4)),
            "realization": list(range(palette_size * 4)),
            "control_name": [f"control_{i}" for i in range(palette_size * 4)],
            "control_value": [i * 0.1 for i in range(palette_size * 4)],
            "Response": list(range(palette_size * 4)),
        }
    )
    plot = EverestGradientsPlot()
    plot.set_selected_controls([f"control_{i}" for i in range(palette_size * 4)])
    figure = create_everest_figure(plot, data, generic_plot_context, everest_ensemble)
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


def test_gradient_plot_on_hover_functionality(
    controls_data, generic_plot_context, everest_ensemble
):
    plot = EverestGradientsPlot()
    plot.set_selected_controls(["control_1"])
    figure = create_everest_figure(
        plot, controls_data, generic_plot_context, everest_ensemble
    )

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
