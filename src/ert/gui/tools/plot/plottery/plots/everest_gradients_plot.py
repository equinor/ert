from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from .plot_tools import ConditionalAxisFormatter, PlotTools

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
        obs_loc: npt.NDArray[np.float32] | None,
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None:
        if not self.selected_controls:
            axes = figure.add_subplot(111)
            axes.text(
                0.5,
                0.5,
                "Select control(s) from the side panel to view gradients",
                ha="center",
                va="center",
            )
            return

        config = plot_context.plotConfig()
        axes = figure.add_subplot(111)

        plot_context.y_axis = plot_context.VALUE_AXIS
        plot_context.x_axis = plot_context.INDEX_AXIS
        plot_context.deactivateDateSupport()

        response_key = key_def.key if key_def else "Response"

        colors = config.lineColorCycle()

        # Collect all data into one frame
        all_frames = []
        for df in ensemble_to_data_map.values():
            if (
                df.empty
                or response_key not in df.columns
                or "control_name" not in df.columns
            ):
                continue
            filtered = df[df["control_name"].isin(self.selected_controls)]

            if not filtered.empty:
                all_frames.append(filtered)

        if not all_frames:
            axes.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        combined = pd.concat(all_frames)
        batch_ids = sorted(combined["batch_id"].unique())

        pos = np.arange(len(self.selected_controls))  # one position per control
        n_batches = len(batch_ids)
        bar_width = 0.8 / n_batches

        for i, batch_id in enumerate(batch_ids):
            color = colors[i % len(colors)][0]
            batch_data = combined[combined["batch_id"] == batch_id]

            values = []
            for control in self.selected_controls:
                match = batch_data[batch_data["control_name"] == control]
                values.append(match[response_key].values[0] if not match.empty else 0.0)

            offsets = pos + (i - n_batches / 2) * bar_width + bar_width / 2
            axes.bar(
                offsets,
                values,
                width=bar_width,
                color=color,
                alpha=0.7,
                label=f"Batch {batch_id}",
            )

        axes.legend()
        figure.tight_layout()


        axes.set_xticks(pos)
        rotation = 0
        if len(self.selected_controls) > 3:
            rotation = 30

        axes.set_xticklabels(self.selected_controls, rotation=rotation)
        # Not sure if this is needed
        axes.yaxis.set_major_formatter(ConditionalAxisFormatter())

        
        PlotTools.finalizePlot(
        plot_context, figure, axes, default_x_label="Control Variable", default_y_label="Gradient"
        )