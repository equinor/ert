from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import polars as pl
from matplotlib.collections import TriMesh
from matplotlib.ticker import EngFormatter

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

MISFIT_MAP_TITLE = "Mean signed χ² Misfit"
OBSERVATIONS_MAP_TITLE = "Observations"
MEAN_SIMULATED_RESPONSE_MAP_TITLE = "Mean Simulated Response"
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

        plot_context.deactivate_date_support()
        if observation_data.empty:
            figure.text(
                0.5,
                0.5,
                "No observation data available",
                ha="center",
                va="center",
                fontsize=12,
            )
            return

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

        if plot_context.mean_simulated_response_map:
            maps.append(
                (
                    MEAN_SIMULATED_RESPONSE_MAP_TITLE,
                    lambda axes: draw_mean_simulated_response_map(
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

        first_axes: Axes | None = None
        gridspec = figure.add_gridspec(1, len(maps))
        for column, (title, draw) in enumerate(maps):
            axes = figure.add_subplot(gridspec[0, column], sharey=first_axes)
            first_axes = first_axes or axes
            draw(axes)
            axes.set_title(title)
            if column > 0:
                axes.tick_params(axis="y", labelleft=False)
                axes.set_ylabel("")

        figure_title = plot_context.plotConfig().title()
        if figure_title is not None:
            figure.suptitle(figure_title)


def draw_observations_map(
    figure: Figure,
    axes: Axes,
    plot_context: PlotContext,
    observation_data: pd.DataFrame,
) -> None:

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
    misfits_by_realization = _preprocess_ensemble_data(
        ensemble_to_data_map, observation_data, axes, "misfit"
    )
    if misfits_by_realization is None:
        return

    east = misfits_by_realization["EAST"].to_numpy()
    north = misfits_by_realization["NORTH"].to_numpy()
    misfit_values = misfits_by_realization["misfit"].to_numpy()

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


def draw_mean_simulated_response_map(
    figure: Figure,
    axes: Axes,
    plot_context: PlotContext,
    ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
    observation_data: pd.DataFrame,
) -> None:

    responses_by_realization = _preprocess_ensemble_data(
        ensemble_to_data_map, observation_data, axes, "response"
    )

    if responses_by_realization is None:
        return

    responses_tripcolor = axes.tripcolor(
        responses_by_realization["EAST"].to_numpy(),
        responses_by_realization["NORTH"].to_numpy(),
        responses_by_realization["response"].to_numpy(),
        shading="gouraud",
        cmap="viridis",
    )

    _finalize_map(figure, axes, responses_tripcolor, plot_context)


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
    axes.xaxis.set_major_formatter(EngFormatter(sep=""))
    axes.yaxis.set_major_formatter(EngFormatter(sep=""))
    axes.set_aspect("equal")
    axes.set_anchor("C")
    PlotTools.finalize_plot(
        plot_context,
        figure,
        axes,
        default_x_label="east coordinate",
        default_y_label="north coordinate",
    )


def _preprocess_ensemble_data(
    ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
    observation_data: pd.DataFrame,
    axes: Axes,
    column: Literal["misfit", "response"],
) -> pl.DataFrame | None:

    if not ensemble_to_data_map:
        PlotTools.show_no_data(axes, "No ensemble data available")
        return None

    if len(ensemble_to_data_map) > 1:
        PlotTools.show_no_data(
            axes, "Multiple ensembles selected; this map supports one at a time"
        )
        return None

    ensemble, ensemble_data = next(iter(ensemble_to_data_map.items()))
    ensemble_by_realization = wide_pandas_to_long_polars_with_misfits(
        {(ensemble.name, ensemble.id): ensemble_data},
        observation_data,
        "seismic",
    )[ensemble.name, ensemble.id]

    if ensemble_by_realization.is_empty():
        PlotTools.show_no_data(axes, f"No {column} data available")
        return None

    return ensemble_by_realization.group_by(["EAST", "NORTH"]).agg(
        pl.col(column).mean()
    )
