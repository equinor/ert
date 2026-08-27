from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    from matplotlib.figure import Figure

    from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.plotting.utils import PlotContext
    from ert.gui.plotting.utils.plot_types import ObservationPlotLocations


class SeismicMapPlot:
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
        axes = figure.add_subplot(111)
        axes.text(
            0.5,
            0.5,
            "Seismic map: not implemented yet",
            ha="center",
            va="center",
        )
        axes.set_axis_off()
