from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ert.gui.plotting.utils.plot_context import PlotType
from ert.gui.plotting.utils.plot_tools import ConditionalAxisFormatter, PlotTools

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.figure import Figure

    from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.plotting.utils import PlotContext
    from ert.gui.plotting.utils.plot_types import ObservationPlotLocations


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
        obs_loc: ObservationPlotLocations | None,
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None:
        axes = figure.add_subplot(111)
        if not self.selected_controls:
            axes.text(
                0.5,
                0.5,
                "Select control(s) from the right side panel to view gradients",
                ha="center",
                va="center",
            )
            return

        config = plot_context.plotConfig()

        plot_context.y_axis = plot_context.VALUE_AXIS
        plot_context.x_axis = plot_context.INDEX_AXIS
        plot_context.plot_type = PlotType.BAR
        plot_context.deactivate_date_support()

        response_key = key_def.key if key_def else "Response"

        # Collect all data into one frame
        all_frames = [
            filtered
            for df in ensemble_to_data_map.values()
            if not df.empty
            and response_key in df.columns
            and "control_name" in df.columns
            and (filtered := df[df["control_name"].isin(self.selected_controls)]).shape[
                0
            ]
        ]

        if not all_frames:
            axes.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        combined = pd.concat(all_frames)
        batch_ids = sorted(combined["batch_id"].unique())

        pos = np.arange(len(batch_ids))
        for idx, x in enumerate(pos):
            if idx % 2 == 0:
                axes.axvspan(x - 0.5, x + 0.5, color="grey", alpha=0.07)
        for edge in np.arange(pos[0] - 0.5, pos[-1] + 1.0, 1.0):
            axes.axvline(edge, color="grey", linewidth=0.8, linestyle="--", alpha=0.4)
        n_controls = len(self.selected_controls)
        bar_width = 0.8 / n_controls

        bar_containers = []
        n_colors = config.get_number_of_colors()
        for i, control in enumerate(self.selected_controls):
            values = []
            for batch_id in batch_ids:
                batch_data = combined[combined["batch_id"] == batch_id]
                match = batch_data[batch_data["control_name"] == control]
                values.append(
                    match[response_key].to_numpy()[0] if not match.empty else 0.0
                )

            offsets = pos + (i - n_controls / 2) * bar_width + bar_width / 2
            color = config.next_color()
            if i < n_colors:
                bar_style = ""
            elif i < n_colors * 2:
                bar_style = "//"
            elif i < n_colors * 3:
                bar_style = ".."
            else:
                bar_style = "-"
            bars = axes.bar(
                offsets,
                values,
                width=bar_width,
                color=color,
                alpha=0.7,
                hatch=bar_style,
            )
            config.add_legend_item(control, bars[0])
            bar_containers.append(bars)

        axes.axhline(0, color="black", linewidth=1.0, alpha=0.3)
        axes.set_xticks(pos)
        axes.set_xticklabels([f"Batch {b}" for b in batch_ids], rotation=0)
        axes.yaxis.set_major_formatter(ConditionalAxisFormatter())

        bar_labels = [
            f"batch {batch}\n{control}"
            for batch in batch_ids
            for control in self.selected_controls
        ]
        PlotTools.labels_on_hover(
            PlotType.BAR, axes, figure, data=bar_containers, labels=bar_labels
        )

        PlotTools.finalizePlot(
            plot_context,
            figure,
            axes,
            default_x_label="",
            default_y_label="Gradient",
        )
