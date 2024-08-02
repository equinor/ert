import io
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import QuadMesh
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
        std_dev_images: Dict[str, bytes],
    ) -> None:
        ensemble_count = len(plot_context.ensembles())
        layer = plot_context.layer
        if layer is not None:
            vmin, vmax = float("inf"), float("-inf")
            axes = []
            images = []

            for i, ensemble in enumerate(plot_context.ensembles(), start=1):
                ax = figure.add_subplot(1, ensemble_count, i)
                axes.append(ax)

                image_data = std_dev_images[ensemble.name]
                if not image_data:
                    ax.set_axis_off()
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for {ensemble.experiment_name} : {ensemble.name}",
                        ha="center",
                        va="center",
                    )
                    images.append(None)
                else:
                    img = plt.imread(io.BytesIO(image_data))
                    images.append(img)
                    vmin = min(vmin, np.min(img))
                    vmax = max(vmax, np.max(img))

                ax.set_title(
                    f"{ensemble.experiment_name} : {ensemble.name} layer={layer}",
                    wrap=True,
                )

            for ax, img in zip(axes, images):
                if img is not None:
                    im = ax.imshow(img, vmin=vmin, vmax=vmax)
                    self._colorbar(im)

            figure.tight_layout()

    @staticmethod
    def _colorbar(mappable: QuadMesh) -> Any:
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
