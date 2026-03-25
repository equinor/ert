from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

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
        axes = figure.add_subplot(111)
        if not self.selected_controls:
            axes.text(
                0.5,
                0.5,
                "Select control(s) from the side panel to view gradients",
                ha="center",
                va="center",
            )
            return

        config = plot_context.plotConfig()

        plot_context.y_axis = plot_context.VALUE_AXIS
        plot_context.x_axis = plot_context.INDEX_AXIS
        plot_context.deactivateDateSupport()

        response_key = key_def.key if key_def else "Response"

        colors = config.lineColorCycle()

        # Collect all data into one frame
        all_frames = []
        for df in ensemble_to_data_map.values():
            if (
                not df.empty
                and response_key in df.columns
                and "control_name" in df.columns
                and (filtered := df[df["control_name"].isin(self.selected_controls)]).shape[0]
            ):
                all_frames.append(filtered)

        if not all_frames:
            axes.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        combined = pd.concat(all_frames)
        batch_ids = sorted(combined["batch_id"].unique())

        pos = np.arange(len(batch_ids))
        n_controls = len(self.selected_controls)
        bar_width = 0.8 / n_controls

        for i, control in enumerate(self.selected_controls):
            color = colors[i % len(colors)][0]
            values = []
            for batch_id in batch_ids:
                batch_data = combined[combined["batch_id"] == batch_id]
                match = batch_data[batch_data["control_name"] == control]
                values.append(
                    match[response_key].to_numpy()[0] if not match.empty else 0.0
                )

            offsets = pos + (i - n_controls / 2) * bar_width + bar_width / 2

            bars = axes.bar(
                offsets,
                values,
                width=bar_width,
                color=color,
                alpha=0.7,
            )
            config.addLegendItem(control, bars[0])
        axes.axhline(0, color="black", linewidth=1.0, alpha=0.3)
        axes.set_xticks(pos)
        axes.set_xticklabels([str(b) for b in batch_ids], rotation=0)
        axes.yaxis.set_major_formatter(ConditionalAxisFormatter())

        PlotTools.finalizePlot(
            plot_context,
            figure,
            axes,
            default_x_label="Batch",
            default_y_label="Gradient",
        )
