from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ert.summary_key_type import is_rate

from .history import plotHistory
from .observations import plotObservations
from .plot_tools import PlotTools

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ert.gui.tools.plot.plot_api import EnsembleObject
    from ert.gui.tools.plot.plottery import PlotConfig, PlotContext


class EnsemblePlot:
    def __init__(self) -> None:
        self.dimensionality = 2

    def plot(
        self,
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_images: dict[str, npt.NDArray[np.float32]],
    ) -> None:
        config = plot_context.plotConfig()
        axes = figure.add_subplot(111)

        plot_context.y_axis = plot_context.VALUE_AXIS
        plot_context.x_axis = plot_context.DATE_AXIS
        draw_style = "steps-pre" if is_rate(plot_context.key()) else None

        for ensemble, data in ensemble_to_data_map.items():
            data = data.T

            if not data.empty:
                if data.index.inferred_type != "datetime64":
                    plot_context.deactivateDateSupport()
                    plot_context.x_axis = plot_context.INDEX_AXIS

                self._plotLines(
                    axes,
                    config,
                    data,
                    f"{ensemble.experiment_name} : {ensemble.name}",
                    draw_style,
                )
                config.nextColor()

        plotObservations(observation_data, plot_context, axes)
        plotHistory(plot_context, axes)

        default_x_label = "Date" if plot_context.isDateSupportActive() else "Index"
        PlotTools.finalizePlot(
            plot_context,
            figure,
            axes,
            default_x_label=default_x_label,
            default_y_label="Value",
        )

    @staticmethod
    def _plotLines(
        axes: Axes,
        plot_config: PlotConfig,
        data: pd.DataFrame,
        ensemble_label: str,
        draw_style: str | None = None,
    ) -> None:
        style = plot_config.defaultStyle()

        if len(data) == 1 and not style.marker:
            style.marker = "."

        lines = axes.plot(
            data.index.to_numpy(),
            data.to_numpy(),
            color=style.color,
            alpha=style.alpha,
            marker=style.marker,
            linewidth=style.width,
            linestyle=style.line_style,
            markersize=style.size,
            drawstyle=draw_style,
        )

        if len(lines) > 0:
            plot_config.addLegendItem(ensemble_label, lines[0])
