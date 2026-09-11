from __future__ import annotations

from typing import TYPE_CHECKING

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


class MisfitMapPlot:
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

        if not ensemble_to_data_map:
            self._show_no_data(figure, "No ensemble data available")
            return

        if len(ensemble_to_data_map) > 1:
            self._show_no_data(
                figure, "Multiple ensembles selected; misfit map supports one at a time"
            )
            return

        if observation_data.empty:
            self._show_no_data(figure, "No observation data available")
            return

        ensemble, ensemble_data = next(iter(ensemble_to_data_map.items()))
        misfits_by_realization = MisfitsPlot._wide_pandas_to_long_polars_with_misfits(
            {(ensemble.name, ensemble.id): ensemble_data},
            observation_data,
            "seismic",
        )[ensemble.name, ensemble.id]

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

        vmin, vmax = (None, None)
        if plot_context.colorbar_range is not None:
            vmin, vmax = plot_context.colorbar_range
        misfit_tripcolor = axes_misfit.tripcolor(
            east,
            north,
            misfit_values,
            shading="flat",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )

        cbar = figure.colorbar(
            misfit_tripcolor,
            ax=axes_misfit,
            label="Mean signed χ²",
            orientation="vertical",
            pad=0.15,
            aspect=40,
        )

        cbar.ax.set_visible(plot_context.plotConfig().is_legend_enabled())
        cbar.ax.ticklabel_format(useOffset=False, style="plain")
        config = plot_context.plotConfig()
        axes_misfit.spines["top"].set_visible(False)
        axes_misfit.spines["right"].set_visible(False)
        axes_misfit.spines["left"].set_visible(False)
        axes_misfit.spines["bottom"].set_visible(False)
        axes_misfit.set_title(config.title())
        axes_misfit.ticklabel_format(useOffset=False, style="plain")
        axes_misfit.set_aspect("equal")
        axes_misfit.set_xlabel(config.x_label() or "east coordinate")
        axes_misfit.set_ylabel(config.y_label() or "north coordinate")
        axes_misfit.grid(config.is_grid_enabled())
        axes_misfit.set_xlim(east.min(), east.max())
        axes_misfit.set_ylim(north.min(), north.max())
