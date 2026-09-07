from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.plotting.utils import PlotContext
    from ert.gui.plotting.utils.plot_types import ObservationPlotLocations


class ObservationsMapPlot:
    def __init__(self) -> None:
        self.dimensionality = 2
        self.requires_observations = True

    @staticmethod
    def _show_no_data(figure: Figure, message: str) -> None:
        axes = figure.add_subplot(111)
        axes.text(0.5, 0.5, message, ha="center", va="center")
        axes.set_axis_off()

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
        if observation_data.empty:
            self._show_no_data(figure, "No observation data available")
            return

        observation = pl.from_pandas(observation_data.T).rename(
            {"EAST": "EAST", "NORTH": "NORTH", "OBS": "OBS"}
        )

        east = observation.get_column("EAST").to_numpy()
        north = observation.get_column("NORTH").to_numpy()
        observation_values = observation.get_column("OBS").to_numpy()

        axes = figure.add_subplot(111)
        observation_tripcolor = axes.tripcolor(
            east, north, observation_values, shading="flat", cmap="viridis"
        )

        cbar = figure.colorbar(
            observation_tripcolor,
            ax=axes,
            label="Observation value",
            orientation="vertical",
            pad=0.15,
            aspect=40,
        )

        cbar.ax.set_visible(plot_context.plotConfig().is_legend_enabled())
        cbar.ax.ticklabel_format(useOffset=False, style="plain")
        config = plot_context.plotConfig()
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.spines["left"].set_visible(False)
        axes.spines["bottom"].set_visible(False)
        axes.set_title(config.title())
        axes.ticklabel_format(useOffset=False, style="plain")
        axes.set_aspect("equal")
        axes.set_xlabel(config.x_label() or "east coordinate")
        axes.set_ylabel(config.y_label() or "north coordinate")
        axes.set_xlim(east.min(), east.max())
        axes.grid(config.is_grid_enabled())
        axes.set_ylim(north.min(), north.max())
