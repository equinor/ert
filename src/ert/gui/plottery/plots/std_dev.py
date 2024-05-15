import io
from typing import Any, Dict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ert.gui.plottery import PlotContext


class StdDevPlot:
    def __init__(self):
        self.dimensionality = 3

    def plot(
        self,
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map,
        _observation_data,
        std_dev_images: Dict[str, bytes],
    ) -> None:
        ensemble_count = len(plot_context.ensembles())
        layer = plot_context.layer
        if layer is not None:
            for i, ensemble_name in enumerate(plot_context.ensembles(), start=1):
                ax = figure.add_subplot(1, ensemble_count, i)
                images = std_dev_images[ensemble_name]
                if not images:
                    ax.set_axis_off()
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for {ensemble_name}",
                        ha="center",
                        va="center",
                    )
                else:
                    img = plt.imread(io.BytesIO(images))
                    ax.imshow(img)
                    ax.set_title(f"{ensemble_name} layer={layer}")
                    p = ax.pcolormesh(img)
                    self._colorbar(p)

    @staticmethod
    def _colorbar(mappable) -> Any:
        # https://joseph-long.com/writing/colorbars/
        last_axes = plt.gca()
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        plt.sca(last_axes)
        return cbar
