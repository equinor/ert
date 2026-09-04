import pandas as pd
from matplotlib.figure import Figure

from ert.gui.plotting.ert_plots.waterfall import (
    COLOR_NEGATIVE,
    COLOR_POSITIVE,
    COLOR_TOTAL,
    WaterfallPlot,
)
from ert.gui.plotting.plot_api import EnsembleObject
from ert.gui.plotting.utils import PlotConfig, PlotContext


def _plot_context(ensemble: EnsembleObject) -> PlotContext:
    return PlotContext(
        PlotConfig(),
        ensembles=[ensemble],
        ensembles_color_indexes=[0],
        key="PARAMETER",
        layer=None,
    )


def test_that_waterfall_plot_works_when_data_is_unavailable() -> None:
    figure = Figure()

    WaterfallPlot().plot(
        figure,
        _plot_context(EnsembleObject("ensemble", "id", False, "experiment", "")),
        ensemble_to_data_map={},
        observation_data=pd.DataFrame(),
        std_dev_images={},
        obs_loc=None,
    )

    assert len(figure.axes) == 0


def test_that_waterfall_plot_stacks_positive_and_negative_contributions() -> None:
    ensemble = EnsembleObject("ensemble", "id", False, "experiment", "")
    figure = Figure()

    WaterfallPlot().plot(
        figure,
        _plot_context(ensemble),
        ensemble_to_data_map={
            ensemble: pd.DataFrame(
                {
                    "type": ["prior", "contribution", "contribution", "posterior"],
                    "name": ["Prior", "OBS_1", "OBS_2", "Posterior"],
                    "value": [1.0, 2.0, -0.5, 2.5],
                }
            )
        },
        observation_data=pd.DataFrame(),
        std_dev_images={},
        obs_loc=None,
    )

    bars = figure.axes[0].patches
    assert [bar.get_height() for bar in bars] == [1.0, 2.0, 0.5, 2.5]
    assert [bar.get_y() for bar in bars] == [0.0, 1.0, 2.5, 0.0]
    assert bars[0].get_facecolor()[:3] == tuple(
        int(COLOR_TOTAL[index : index + 2], 16) / 255 for index in (1, 3, 5)
    )
    assert bars[1].get_facecolor()[:3] == tuple(
        int(COLOR_POSITIVE[index : index + 2], 16) / 255 for index in (1, 3, 5)
    )
    assert bars[2].get_facecolor()[:3] == tuple(
        int(COLOR_NEGATIVE[index : index + 2], 16) / 255 for index in (1, 3, 5)
    )
    assert bars[3].get_facecolor() == bars[0].get_facecolor()
