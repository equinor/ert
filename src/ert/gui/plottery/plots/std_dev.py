from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ert.gui.plottery import PlotContext
from ert.gui.tools.plot.plot_api import EnsembleObject


class StdDevPlot:
    def __init__(self) -> None:
        self.dimensionality = 3

    def plot(
        self,
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: Dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_data: Dict[str, npt.NDArray[np.float32]],
    ) -> None:
        ensemble_count = len(plot_context.ensembles())
        layer = plot_context.layer
        if layer is not None:
            vmin: float = np.inf
            vmax: float = -np.inf
            axes = []
            images: List[npt.NDArray[np.float32]] = []
            for i, ensemble in enumerate(plot_context.ensembles(), start=1):
                ax = figure.add_subplot(1, ensemble_count, i)
                axes.append(ax)
                data = std_dev_data[ensemble.name]
                if data.size == 0:
                    ax.set_axis_off()
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for {ensemble.experiment_name} : {ensemble.name}",
                        ha="center",
                        va="center",
                    )
                else:
                    images.append(data)
                    vmin = min(vmin, float(np.min(data)))
                    vmax = max(vmax, float(np.max(data)))
                ax.set_title(
                    f"{ensemble.experiment_name} : {ensemble.name} layer={layer}",
                    wrap=True,
                )

            norm = plt.Normalize(vmin, vmax)
            for ax, data in zip(axes, images):
                if data is not None:
                    im = ax.imshow(data, norm=norm, cmap="viridis")
                    self._colorbar(im)
            figure.tight_layout()

    @staticmethod
    def _colorbar(mappable: Any) -> Any:
        # https://joseph-long.com/writing/colorbars/
        last_axes = plt.gca()
        ax = mappable.axes
        assert ax is not None
        fig = ax.figure
        assert fig is not None
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        plt.sca(last_axes)
        return cbar
