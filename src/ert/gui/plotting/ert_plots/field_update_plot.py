from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from ert.config.field import Field
from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.plotting.utils.plot_types import ObservationPlotLocations
from ert.gui.utils import truncate_experiment_name

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ert.gui.plotting.utils import PlotContext


class FieldUpdatePlot:
    def __init__(self) -> None:
        self.dimensionality = 3
        self.requires_observations = False

    @staticmethod
    def plot(
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_images: dict[str, npt.NDArray[np.float32]],
        obs_loc: ObservationPlotLocations | None,
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None:
        axes = figure.add_subplot(111)
        ensembles = plot_context.ensembles()
        if len(ensembles) != 2:
            axes.text(
                0.5,
                0.5,
                "Select exactly two ensembles",
                ha="center",
                va="center",
            )
            return

        if not isinstance(key_def.parameter if key_def else None, Field):
            axes.text(0.5, 0.5, "Select a FIELD parameter", ha="center", va="center")
            return

        if plot_context.layer is None:
            axes.text(0.5, 0.5, "Select a field layer", ha="center", va="center")
            return

        field = key_def.parameter
        prior, posterior = ensembles
        try:
            prior_values = _layer_mean(
                ensemble_to_data_map[prior], field, plot_context.layer
            )
            posterior_values = _layer_mean(
                ensemble_to_data_map[posterior], field, plot_context.layer
            )
        except (KeyError, ValueError):
            axes.text(0.5, 0.5, "No field data available", ha="center", va="center")
            return

        difference = posterior_values - prior_values
        limit = float(np.nanmax(np.abs(difference)))
        if limit == 0.0:
            limit = 1.0

        image = axes.imshow(
            difference,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="equal",
        )
        if obs_loc is not None:
            axes.scatter(
                obs_loc.x - 0.5,
                obs_loc.y - 0.5,
                c="black",
                marker="o",
                edgecolors="white",
                linewidths=0.5,
            )

        prior_label = (
            f"{truncate_experiment_name(prior.experiment_name)} : {prior.name}"
        )
        posterior_label = (
            f"{truncate_experiment_name(posterior.experiment_name)} : {posterior.name}"
        )
        axes.set_title(
            f"{plot_context.key()} layer={plot_context.layer}\n"
            f"{posterior_label} mean - {prior_label} mean"
        )
        figure.colorbar(image, ax=axes, label="Mean update")


def _layer_mean(
    field_data: pd.DataFrame, field: Field, layer: int
) -> npt.NDArray[np.float64]:
    values = field_data.to_numpy(dtype=float)
    if values.shape == (field.nx, field.ny):
        return values

    values = values.reshape(-1)
    expected_size = field.nx * field.ny * field.nz
    if values.size != expected_size:
        raise ValueError("Field mean has incompatible dimensions")
    return values.reshape(field.nx, field.ny, field.nz)[:, :, layer]
