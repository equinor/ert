from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

if TYPE_CHECKING:
    from ert.gui.tools.plot.plot_api import EnsembleObject
    from ert.gui.tools.plot.plottery import PlotContext


class StdDevPlot:
    def __init__(self) -> None:
        self.dimensionality = 3

    def plot(
        self,
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_data: dict[str, npt.NDArray[np.float32]],
    ) -> None:
        ensemble_count = len(plot_context.ensembles())
        layer = plot_context.layer
        if layer is not None:
            vmin: float = np.inf
            vmax: float = -np.inf
            heatmaps = []
            boxplot_axes = []

            # Adjust height_ratios to reduce space between plots
            figure.set_layout_engine("constrained")
            gridspec = figure.add_gridspec(2, ensemble_count, hspace=0.2)

            for i, ensemble in enumerate(plot_context.ensembles(), start=1):
                ax_heat = figure.add_subplot(gridspec[0, i - 1])
                ax_box = figure.add_subplot(gridspec[1, i - 1])
                data = std_dev_data[ensemble.name]
                if data.size == 0:
                    ax_heat.set_axis_off()
                    ax_box.set_axis_off()
                    ax_heat.text(
                        0.5,
                        0.5,
                        f"No data for {ensemble.experiment_name} : {ensemble.name}",
                        ha="center",
                        va="center",
                    )
                else:
                    vmin = min(vmin, float(np.min(data)))
                    vmax = max(vmax, float(np.max(data)))

                    im = ax_heat.imshow(data, cmap="viridis", aspect="equal")
                    heatmaps.append(im)

                    ax_box.boxplot(data.flatten(), orientation="vertical", widths=0.5)
                    boxplot_axes.append(ax_box)

                    min_value = np.min(data)
                    mean_value = np.mean(data)
                    max_value = np.max(data)

                    annotation_text = (
                        f"Min: {min_value:.2f}\n"
                        f"Mean: {mean_value:.2f}\nMax: {max_value:.2f}"
                    )
                    ax_box.annotate(
                        annotation_text,
                        xy=(1, 1),  # Changed from (0, 1) to (1, 1)
                        xycoords="axes fraction",
                        ha="right",  # Changed from 'left' to 'right'
                        va="top",
                        fontsize=8,
                        fontweight="bold",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "black",
                            "boxstyle": "round,pad=0.2",
                        },
                    )

                    ax_box.spines["top"].set_visible(False)
                    ax_box.spines["right"].set_visible(False)
                    ax_box.spines["bottom"].set_visible(False)
                    ax_box.spines["left"].set_visible(True)

                    ax_box.set_xticks([])
                    ax_box.set_xticklabels([])

                    ax_heat.set_ylabel("")
                    ax_box.set_ylabel(
                        "Standard Deviation", fontsize=8
                    )  # Reduced font size

                    self._colorbar(im)

                ax_heat.set_title(
                    f"{ensemble.experiment_name} : {ensemble.name} layer={layer}",
                    wrap=True,
                    fontsize=10,  # Reduced font size
                )

            norm = plt.Normalize(vmin, vmax)
            for im in heatmaps:
                im.set_norm(norm)

            padding = 0.05 * (vmax - vmin)
            if padding > 0.0:
                for ax_box in boxplot_axes:
                    ax_box.set_ylim(vmin - padding, vmax + padding)

    @staticmethod
    def _colorbar(mappable: Any) -> Any:
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
