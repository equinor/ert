from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from matplotlib.figure import Figure

    from ert.gui.tools.plot.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.tools.plot.plottery import PlotContext


class EverestGradientsPlot:
    def __init__(self) -> None:
        self.dimensionality = 2
        self.requires_observations = False
        self.selected_controls: list[str] = []

    def set_selected_controls(self, controls: list[str]) -> None:
        self.selected_controls = controls

    def plot(
        self,
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_images: dict[str, npt.NDArray[np.float32]],
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None:
        if not self.selected_controls:
            ax = figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Select control(s) from the side panel to view gradients",
                ha="center",
                va="center",
            )
            return

        config = plot_context.plotConfig()

        n_controls = len(self.selected_controls)
        axs = figure.subplots(n_controls, 1, sharex=True, squeeze=False)

        plot_context.y_axis = plot_context.VALUE_AXIS
        plot_context.x_axis = plot_context.INDEX_AXIS
        plot_context.deactivateDateSupport()

        response_key = key_def.key if key_def else "Response"
        config.setTitle(f"Derivative of {response_key} wrt controls")

        default_color = config.lineColorCycle()[0][0]

        for i, control in enumerate(self.selected_controls):
            ax = axs[i, 0]
            ax.set_ylabel(control)

            ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="gray", alpha=0.5)
            ax.xaxis.grid(False)

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # (derivative is interesting to see go towards/away from 0)
            # Hence thicker horizontal line at y=0
            ax.axhline(0, color="black", linewidth=1.0, alpha=0.3)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if i < n_controls - 1:
                # Intermediate plots: thin gray line as separator
                ax.spines["bottom"].set_visible(True)
                ax.spines["bottom"].set_linewidth(0.5)
                ax.spines["bottom"].set_color("lightgray")
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            else:
                ax.spines["bottom"].set_visible(True)
                ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)

            # For normalizing y-axis across controls
            max_abs_val = 0.0

            data_frames = []
            for df in ensemble_to_data_map.values():
                if (
                    df.empty
                    or response_key not in df.columns
                    or "control_name" not in df.columns
                ):
                    continue

                control_data = df[df["control_name"] == control]
                if not control_data.empty:
                    data_frames.append(control_data)

            if data_frames:
                r_data = pd.concat(data_frames).sort_values("batch_id")

                current_max_abs = r_data[response_key].abs().max()
                max_abs_val = max(max_abs_val, current_max_abs)

                ax.plot(
                    r_data["batch_id"].to_numpy(),
                    r_data[response_key].to_numpy(),
                    "-o",
                    markersize=5,
                    linewidth=2,
                    color=default_color,
                    markerfacecolor=default_color,
                )

                if max_abs_val > 0:
                    limit = max_abs_val * 1.1  # Add some padding
                    ax.set_ylim(-limit, limit)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")

        axs[-1, 0].set_xlabel("Iteration")

        # Adjust layout to remove gaps if removing spines looks like 'sparklines'
        figure.subplots_adjust(hspace=0.0)
        figure.suptitle(f"Derivative of {response_key} wrt controls")
