from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from matplotlib.figure import Figure

from ert.gui.plotting.ert_plots.misfits import MisfitsPlot

if TYPE_CHECKING:
    from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.plotting.utils import PlotContext
    from ert.gui.plotting.utils.plot_types import ObservationPlotLocations


class SeismicMapPlot:
    def __init__(self) -> None:
        self.dimensionality = 2
        self.requires_observations = True
        self._misfit_range_cache: dict[str, float] = {}

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

        if not ensemble_to_data_map:
            self._show_no_data(figure, "No ensemble data available")
            return

        first_ensemble, first_data = next(iter(ensemble_to_data_map.items()))
        misfits_by_realization = MisfitsPlot._wide_pandas_to_long_polars_with_misfits(
            {(first_ensemble.name, first_ensemble.id): first_data},
            observation_data,
            "seismic",
        )[first_ensemble.name, first_ensemble.id]

        if misfits_by_realization.is_empty():
            self._show_no_data(figure, "No misfit data available")
            return

        mean_misfits = misfits_by_realization.group_by(["EAST", "NORTH"]).agg(
            pl.col("misfit").mean()
        )
        east = mean_misfits["EAST"].to_numpy()
        north = mean_misfits["NORTH"].to_numpy()
        misfit_values = mean_misfits["misfit"].to_numpy()
        axes_misfit = figure.add_subplot(111)
        key = key_def.key if key_def is not None else ""
        current_vabs = (
            float(np.max(np.abs(misfit_values))) if misfit_values.size else 0.0
        )
        vabs = max(self._misfit_range_cache.get(key, 0.0), current_vabs)
        if math.isclose(vabs, 0.0):
            vabs = 1.0
        self._misfit_range_cache[key] = vabs

        norm = mcolors.Normalize(vmin=-vabs, vmax=vabs)
        misfit_tripcolor = axes_misfit.tripcolor(
            east, north, misfit_values, shading="flat", cmap="viridis", norm=norm
        )

        figure.colorbar(
            misfit_tripcolor,
            ax=axes_misfit,
            label="Mean signed χ²",
            orientation="horizontal",
            pad=0.15,
            aspect=40,
        )

        axes_misfit.set_title("Misfit map")
        axes_misfit.ticklabel_format(useOffset=True, style="plain")
        axes_misfit.set_aspect("equal")
        axes_misfit.set_xlabel("east coordinate")
        axes_misfit.set_ylabel("north coordinate")
        axes_misfit.set_xlim(east.min(), east.max())
        axes_misfit.set_ylim(north.min(), north.max())
