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
        misfit_tripcolor = axes.tripcolor(
            east, north, observation_values, shading="flat", cmap="viridis"
        )

        figure.colorbar(
            misfit_tripcolor,
            ax=axes,
            label="Observation value",
            orientation="horizontal",
            pad=0.15,
            aspect=40,
        )

        axes.set_title("Observation map")
        axes.ticklabel_format(useOffset=False, style="plain")
        axes.set_aspect("equal")
        axes.set_xlabel("east coordinate")
        axes.set_ylabel("north coordinate")
        axes.set_xlim(east.min(), east.max())
        axes.set_ylim(north.min(), north.max())
