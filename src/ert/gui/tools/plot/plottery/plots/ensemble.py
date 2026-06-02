from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from resfo_utilities import is_rate

from ert.gui.tools.plot.plottery.plot_context import PlotType
from ert.gui.utils import is_everest_application

from .history import plotHistory
from .observations import plotObservations
from .plot_tools import PlotTools

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ert.gui.tools.plot.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.tools.plot.plot_types import ObservationPlotLocations
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
        obs_loc: ObservationPlotLocations | None,
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None:
        is_everest = is_everest_application()
        config = plot_context.plotConfig()
        axes = figure.add_subplot(111)
        if len(ensemble_to_data_map) == 0:
            axes.text(
                0.5,
                0.5,
                f"Select {'batches' if is_everest else 'ensembles'}"
                f" from the right side panel",
                ha="center",
                va="center",
            )
            return

        plot_context.y_axis = plot_context.VALUE_AXIS
        plot_context.x_axis = plot_context.DATE_AXIS
        plot_context.plot_type = PlotType.LINE
        draw_style = "steps-pre" if is_rate(plot_context.key()) else None
        zorder = 0
        tooltip_data = []
        tooltip_labels = []
        for (ensemble, untransposed_data), color_index in zip(
            ensemble_to_data_map.items(),
            plot_context.ensembles_color_indexes(),
            strict=False,
        ):
            data = untransposed_data.T

            if not data.empty:
                if data.index.inferred_type != "datetime64":
                    plot_context.deactivate_date_support()
                    plot_context.x_axis = plot_context.INDEX_AXIS
                config.set_current_color(color_index)
                label = (
                    f"{ensemble.name}"
                    if is_everest
                    else f"{ensemble.experiment_name} : {ensemble.name}"
                )
                lines = self._plotLines(
                    axes,
                    config,
                    data,
                    label,
                    draw_style,
                    zorder=zorder,
                )
                tooltip_data.append(lines)
                tooltip_labels.append(label)
                zorder -= 1

        plotObservations(observation_data, plot_context, axes)
        plotHistory(plot_context, axes)

        default_x_label = "Date" if plot_context.is_date_support_active() else "Index"

        PlotTools.labels_on_hover(
            PlotType.LINE,
            axes,
            figure,
            data=tooltip_data,
            labels=tooltip_labels,
        )

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
        zorder: float = 1,
    ) -> list[Line2D]:
        style = plot_config.default_style()

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
            plot_config.add_legend_item(ensemble_label, lines[0])

        return lines
