from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl
from matplotlib.collections import TriMesh

from ert.gui.plotting.ert_plots.misfits import wide_pandas_to_long_polars_with_misfits
from ert.gui.plotting.utils.plot_tools import PlotTools

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.plotting.utils import PlotContext
    from ert.gui.plotting.utils.plot_types import ObservationPlotLocations

MISFIT_MAP_TITLE = "Mean signed χ² Misfit Map"
OBSERVATIONS_MAP_TITLE = "Observations Map"
NO_MAP_SELECTED_MESSAGE = (
    "No map selected."
    "\n\nFrom the Map options (on the right-side panel),"
    "\nplease select at least one of the map options."
)


class MapsPlot:
    def __init__(self) -> None:
        self.dimensionality = 2
        self.requires_observations = True

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
        maps: list[tuple[str, Callable[[Axes], None]]] = []
        if plot_context.observations_map:
            maps.append(
                (
                    OBSERVATIONS_MAP_TITLE,
                    lambda axes: draw_observations_map(
                        figure, axes, plot_context, observation_data
                    ),
                )
            )

        if plot_context.misfit_map:
            maps.append(
                (
                    MISFIT_MAP_TITLE,
                    lambda axes: draw_misfit_map(
                        figure,
                        axes,
                        plot_context,
                        ensemble_to_data_map,
                        observation_data,
                    ),
                )
            )

        if not maps:
            figure.text(
                0.5,
                0.5,
                NO_MAP_SELECTED_MESSAGE,
                ha="center",
                va="center",
                fontsize=12,
            )
            return

        gridspec = figure.add_gridspec(1, len(maps))
        for column, (title, draw) in enumerate(maps):
            axes = figure.add_subplot(gridspec[0, column])
            draw(axes)
            axes.set_title(title)

        figure_title = plot_context.plotConfig().title()
        if figure_title is not None:
            figure.suptitle(figure_title)


def draw_observations_map(
    figure: Figure,
    axes: Axes,
    plot_context: PlotContext,
    observation_data: pd.DataFrame,
) -> None:
    if observation_data.empty:
        PlotTools.show_no_data(axes, "No observation data available")
        return

    observation = pl.from_pandas(observation_data.T)
    east = observation.get_column("EAST").to_numpy()
    north = observation.get_column("NORTH").to_numpy()
    observation_values = observation.get_column("OBS").to_numpy()

    observation_tripcolor = axes.tripcolor(
        east, north, observation_values, shading="gouraud", cmap="viridis"
    )
    _finalize_map(figure, axes, observation_tripcolor, plot_context)


def draw_misfit_map(
    figure: Figure,
    axes: Axes,
    plot_context: PlotContext,
    ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
    observation_data: pd.DataFrame,
) -> None:
    if not ensemble_to_data_map:
        PlotTools.show_no_data(axes, "No ensemble data available")
        return

    if len(ensemble_to_data_map) > 1:
        PlotTools.show_no_data(
            axes, "Multiple ensembles selected; misfit map supports one at a time"
        )
        return

    if observation_data.empty:
        PlotTools.show_no_data(axes, "No observation data available")
        return

    ensemble, ensemble_data = next(iter(ensemble_to_data_map.items()))
    misfits_by_realization = wide_pandas_to_long_polars_with_misfits(
        {(ensemble.name, ensemble.id): ensemble_data},
        observation_data,
        "seismic",
    )[ensemble.name, ensemble.id]

    if misfits_by_realization.is_empty():
        PlotTools.show_no_data(axes, "No misfit data available")
        return

    mean_misfits = misfits_by_realization.group_by(["EAST", "NORTH"]).agg(
        pl.col("misfit").mean()
    )
    east = mean_misfits["EAST"].to_numpy()
    north = mean_misfits["NORTH"].to_numpy()
    misfit_values = mean_misfits["misfit"].to_numpy()

    vmin, vmax = (None, None)
    if plot_context.colorbar_range is not None:
        vmin, vmax = plot_context.colorbar_range
    misfit_tripcolor = axes.tripcolor(
        east,
        north,
        misfit_values,
        shading="gouraud",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    _finalize_map(figure, axes, misfit_tripcolor, plot_context)


def _finalize_map(
    figure: Figure, axes: Axes, tripcolor: TriMesh, plot_context: PlotContext
) -> None:
    cbar = figure.colorbar(
        tripcolor,
        ax=axes,
        orientation="horizontal",
        aspect=40,
    )

    cbar.ax.set_visible(plot_context.plotConfig().is_legend_enabled())
    cbar.ax.ticklabel_format(useOffset=False, style="plain")
    axes.ticklabel_format(useOffset=False, style="plain")
    axes.set_aspect("equal")
    axes.set_anchor("C")
    PlotTools.finalize_plot(
        plot_context,
        figure,
        axes,
        default_x_label="east coordinate",
        default_y_label="north coordinate",
    )
