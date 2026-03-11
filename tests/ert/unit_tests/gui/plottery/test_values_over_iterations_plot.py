from unittest.mock import Mock

import pandas as pd
import pytest
from matplotlib.figure import Figure

from ert.gui.tools.plot.plot_api import EnsembleObject
from ert.gui.tools.plot.plottery import PlotConfig, PlotContext
from ert.gui.tools.plot.plottery.plots.values_over_iteration_plot import (
    ValuesOverIterationsPlot,
)

THRESHOLD = ValuesOverIterationsPlot.LEGEND_THRESHOLD


def make_improvement_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batch_id": list(range(2)),
            "realization": [0] * 2,
            "value": [1.0, 0.8],
            "is_improvement": [True, False],
        }
    )


def make_realization_df(n_realizations: int) -> pd.DataFrame:
    rows = [
        {"batch_id": b, "realization": r, "value": float(r + b)}
        for r in range(n_realizations)
        for b in range(2)
    ]
    return pd.DataFrame(rows)


def setup_plotting(dataframe: pd.DataFrame) -> ValuesOverIterationsPlot:
    context = Mock(spec=PlotContext)
    context.plotConfig.return_value = PlotConfig(title="Values Over Iterations")
    ensemble = EnsembleObject(
        "ensemble_1", "id", False, "experiment_1", started_at="2024-01-01T00:00:00"
    )
    plotter = ValuesOverIterationsPlot()
    plotter.plot(
        Figure(),
        context,
        {ensemble: dataframe},
        pd.DataFrame(),
        {},
        None,
    )
    return plotter


def legend_texts(plotter: ValuesOverIterationsPlot) -> list[str]:
    axes = plotter._axes
    assert axes is not None
    legend = axes.get_legend()
    return [t.get_text() for t in legend.get_texts()] if legend else []


@pytest.mark.parametrize("is_improvement", [True, False])
def test_that_is_improvement_data_inhibit_hover_labels(is_improvement: bool) -> None:
    plotter = setup_plotting(make_improvement_df())

    assert plotter.is_improvement is True
    assert "Accepted" in legend_texts(plotter)
    assert "Rejected" in legend_texts(plotter)

    texts_before = legend_texts(plotter)
    plotter.is_improvement = is_improvement
    plotter._legend_count = THRESHOLD + 1  # ensure count alone would trigger update
    plotter.update_legend(plotter._axes.get_lines()[0])  # type: ignore[union-attr]

    assert (legend_texts(plotter) == texts_before) == is_improvement


def test_that_update_legend_sets_axes_legend_when_below_or_at_threshold() -> None:
    for count in [THRESHOLD - 1, THRESHOLD]:
        plotter = setup_plotting(make_realization_df(count))
        assert len(legend_texts(plotter)) == count
        plotter.update_legend(plotter._axes.get_lines()[0])  # type: ignore[union-attr]
        assert len(legend_texts(plotter)) == count


def test_that_update_legend_sets_axes_legend_when_above_threshold() -> None:
    plotter = setup_plotting(make_realization_df(THRESHOLD + 1))
    assert plotter._axes.get_legend() is None  # type: ignore[union-attr]
    plotter.update_legend(plotter._axes.get_lines()[0])  # type: ignore[union-attr]
    line = plotter._axes.get_lines()[0]  # type: ignore[union-attr]
    assert legend_texts(plotter) == [line.get_label()]
