from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.utils import truncate_experiment_name

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.figure import Figure

    from ert.gui.plotting.utils import PlotContext
    from ert.gui.plotting.utils.plot_types import ObservationPlotLocations

COLOR_TOTAL = "#4472C4"
COLOR_POSITIVE = "#70AD47"
COLOR_NEGATIVE = "#ED7D31"


class WaterfallPlot:
    def __init__(self) -> None:
        self.dimensionality = 1
        self.requires_observations = False

    @staticmethod
    def plot(
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_images: dict[str, npt.NDArray[np.float32]],
        obs_loc: ObservationPlotLocations | None,
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None:
        if not ensemble_to_data_map:
            return

        # Find the first ensemble that has non-empty waterfall data
        ensemble = None
        data = pd.DataFrame()
        for ens, df in ensemble_to_data_map.items():
            if not df.empty:
                ensemble, data = ens, df
                break

        if data.empty or ensemble is None:
            ax = figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No waterfall data available.\n"
                "This plot requires an ensemble produced by EnIF\n"
                "and a scalar (gen_kw) parameter.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.set_axis_off()
            return

        _plot_waterfall(figure, plot_context, data, ensemble)


def _plot_waterfall(
    figure: Figure,
    plot_context: PlotContext,
    data: pd.DataFrame,
    ensemble: EnsembleObject,
) -> None:
    """Render a waterfall chart from pre-computed contribution data.

    Expected DataFrame columns:
        type: "prior" | "contribution" | "posterior"
        name: label for the bar
        value: contribution value (or absolute value for prior/posterior)
    """
    types = data["type"].to_numpy()
    names = data["name"].to_list()
    values = data["value"].to_numpy().astype(float)

    n_bars = len(values)
    x = np.arange(n_bars)

    prior_mask = types == "prior"
    posterior_mask = types == "posterior"
    contrib_mask = types == "contribution"

    prior_mean = float(values[prior_mask][0]) if prior_mask.any() else 0.0
    posterior_mean = float(values[posterior_mask][0]) if posterior_mask.any() else 0.0
    contributions = values[contrib_mask]

    # Compute bar heights and bottoms
    bottoms = np.zeros(n_bars)
    heights = np.zeros(n_bars)

    # First bar: prior
    heights[0] = prior_mean
    bottoms[0] = 0.0

    # Contribution bars
    running = prior_mean
    contrib_idx = 0
    for i in range(n_bars):
        if types[i] == "prior":
            continue
        if types[i] == "posterior":
            bottoms[i] = 0.0
            heights[i] = posterior_mean
            break
        c = contributions[contrib_idx]
        bottoms[i] = running if c >= 0 else running + c
        heights[i] = abs(c)
        running += c
        contrib_idx += 1

    # Bar colors
    bar_colors = []
    for i in range(n_bars):
        if types[i] in {"prior", "posterior"}:
            bar_colors.append(COLOR_TOTAL)
        elif values[i] >= 0:
            bar_colors.append(COLOR_POSITIVE)
        else:
            bar_colors.append(COLOR_NEGATIVE)

    ax = figure.add_subplot(111)
    bars = ax.bar(
        x,
        heights,
        bottom=bottoms,
        color=bar_colors,
        edgecolor="grey",
        linewidth=0.5,
        width=0.7,
    )

    # Connector lines between bars
    for i in range(n_bars - 1):
        top_i = bottoms[i] + heights[i]
        ax.plot(
            [x[i] + 0.35, x[i + 1] - 0.35],
            [top_i, top_i],
            color="grey",
            linewidth=0.6,
            linestyle="--",
        )

    # Value labels on bars
    for i, _bar in enumerate(bars):
        is_total = types[i] in {"prior", "posterior"}
        val = heights[i] if is_total else values[i]
        label_y = bottoms[i] + heights[i]
        va = "bottom"
        offset = 2
        if not is_total and val < 0:
            va = "top"
            label_y = bottoms[i]
            offset = -2
        label = f"{heights[i]:.4f}" if is_total else f"{val:+.4f}"
        ax.annotate(
            label,
            xy=(x[i], label_y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean parameter value (standardized)", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    key = plot_context.key()
    experiment_name = truncate_experiment_name(ensemble.experiment_name)
    n_contrib = int(contrib_mask.sum())
    ax.set_title(
        f"Expected update for {key}\n"
        f"by top {n_contrib} observation contributions"
        f"\n({experiment_name} : {ensemble.name})",
        fontsize=12,
        fontweight="bold",
    )

    figure.tight_layout()
