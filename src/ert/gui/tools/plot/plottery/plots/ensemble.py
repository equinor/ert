from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from resfo_utilities import is_rate

from .history import plotHistory
from .observations import plotObservations
from .plot_tools import PlotTools

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ert.gui.tools.plot.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.tools.plot.plottery import PlotConfig, PlotContext


class EnsemblePlot:
    def __init__(self) -> None:
        self.dimensionality = 2
        self.requires_observations = False

    def plot(
        self,
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_images: dict[str, npt.NDArray[np.float32]],
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None:
        config = plot_context.plotConfig()
        axes = figure.add_subplot(111)

        plot_context.y_axis = plot_context.VALUE_AXIS
        plot_context.x_axis = plot_context.DATE_AXIS
        draw_style = "steps-pre" if is_rate(plot_context.key()) else None
        zorder = 0
        for (ensemble, data), color_index in zip(
            ensemble_to_data_map.items(),
            plot_context.ensembles_color_indexes(),
            strict=False,
        ):
            data = data.T

            if not data.empty:
                if data.index.inferred_type != "datetime64":
                    plot_context.deactivateDateSupport()
                    plot_context.x_axis = plot_context.INDEX_AXIS
                config.setCurrentColor(color_index)
                self._plotLines(
                    axes,
                    config,
                    data,
                    f"{ensemble.experiment_name} : {ensemble.name}",
                    draw_style,
                    zorder=zorder,
                )
                zorder -= 1

        observation_bars = plotObservations(observation_data, plot_context, axes)
        plotHistory(plot_context, axes)

        default_x_label = "Date" if plot_context.isDateSupportActive() else "Index"
        PlotTools.finalizePlot(
            plot_context,
            figure,
            axes,
            default_x_label=default_x_label,
            default_y_label="Value",
        )

        def hover(event):
            if not observation_bars:
                return

            observation_lines = observation_bars.lines[2][0]
            observation_dots = observation_bars.lines[0]
            contains_cursor, index = observation_lines.contains(event)
            if contains_cursor:
                annotation.set_visible(True)
                index = int(index["ind"][0])
                x, y = observation_dots.properties()["xydata"][index]
                annotation.xy = (x, y)
            else:
                annotation.set_visible(False)

            figure.canvas.draw_idle()

        annotation = axes.annotate(
            "Test",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "w"},
        )
        annotation.set_visible(False)
        figure.canvas.mpl_connect("motion_notify_event", hover)

    @staticmethod
    def _plotLines(
        axes: Axes,
        plot_config: PlotConfig,
        data: pd.DataFrame,
        ensemble_label: str,
        draw_style: str | None = None,
        zorder: float = 1,
    ) -> None:
        style = plot_config.defaultStyle()

        if len(data) == 1 and not style.marker:
            style.marker = "."

        if plot_config.flip_response_axis:
            x = data.to_numpy()
            y = data.index.to_numpy()
            axes.yaxis.set_inverted(True)
        else:
            y = data.to_numpy()
            x = data.index.to_numpy()
        lines = axes.plot(
            x,
            y,
            color=style.color,
            alpha=style.alpha,
            marker=style.marker,
            linewidth=style.width,
            linestyle=style.line_style,
            markersize=style.size,
            drawstyle=draw_style,
            zorder=zorder,
        )

        if len(lines) > 0:
            plot_config.addLegendItem(ensemble_label, lines[0])
