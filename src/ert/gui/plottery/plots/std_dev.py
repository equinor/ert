from typing import Any, Dict

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
            heatmaps = []
            boxplot_axes = []

            # Adjusted gridspec to allocate less space to boxplots
            gridspec = figure.add_gridspec(2, ensemble_count, height_ratios=[3, 1])

            for i, ensemble in enumerate(plot_context.ensembles(), start=1):
                # Create two subplots: one for heatmap, one for boxplot
                ax_heat = figure.add_subplot(
                    gridspec[0, i - 1]
                )  # Heatmap in the upper row
                ax_box = figure.add_subplot(
                    gridspec[1, i - 1]
                )  # Boxplot in the lower row

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

                    # Create heatmap with aspect='equal' to make it quadratic
                    im = ax_heat.imshow(data, cmap="viridis", aspect="equal")
                    heatmaps.append(im)

                    # Create boxplot
                    ax_box.boxplot(data.flatten(), vert=False, widths=0.5)
                    boxplot_axes.append(ax_box)

                    # Calculate min, mean, and max
                    min_value = np.min(data)
                    mean_value = np.mean(data)
                    max_value = np.max(data)

                    # Create a combined annotation in the top-right corner
                    annotation_text = f"Min: {min_value:.2f}\nMean: {mean_value:.2f}\nMax: {max_value:.2f}"
                    ax_box.annotate(
                        annotation_text,
                        xy=(1, 1),  # Top-right corner
                        xycoords="axes fraction",
                        ha="right",
                        va="top",
                        fontsize=10,
                        fontweight="bold",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "black",
                            "boxstyle": "round,pad=0.3",
                        },
                    )

                    # Remove the frame around the boxplot, except for the bottom spine
                    ax_box.spines["top"].set_visible(False)
                    ax_box.spines["right"].set_visible(False)
                    ax_box.spines["left"].set_visible(False)
                    ax_box.spines["bottom"].set_visible(True)

                    # Remove y-axis ticks and labels
                    ax_box.set_yticks([])
                    ax_box.set_yticklabels([])

                    # Only show xlabel for the bottom subplot
                    ax_heat.set_xlabel("")
                    ax_box.set_xlabel("Standard Deviation")

                    # Add colorbar using _colorbar method
                    self._colorbar(im)

                ax_heat.set_title(
                    f"{ensemble.experiment_name} : {ensemble.name} layer={layer}",
                    wrap=True,
                )

            # Apply normalization to all heatmaps after determining global vmin and vmax
            norm = plt.Normalize(vmin, vmax)
            for im in heatmaps:
                im.set_norm(norm)

            # Set x-limits for all boxplots with padding
            padding = 0.05 * (vmax - vmin)  # 5% padding on each side
            for ax_box in boxplot_axes:
                ax_box.set_xlim(vmin - padding, vmax + padding)

            figure.tight_layout()

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
