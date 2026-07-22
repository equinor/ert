from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.typing import ColorType

from ert.gui.plotting.utils.plot_context import PlotContext


def generate_plots(
    plot_context: PlotContext,
    axes: Axes,
    data: Sequence[npt.NDArray[np.float64]],
    positions: Sequence[float],
    box_width: float,
    color: ColorType,
    lower_percentile: int,
    upper_percentile: int,
    legend_label: str,
) -> None:
    config = plot_context.plotConfig()
    outlier = plot_context.outliers
    scatter = plot_context.scatter_plot
    box = plot_context.box_plot
    mean = plot_context.mean
    if box:
        axes.boxplot(
            data,
            positions=positions,
            widths=box_width,
            whis=(
                lower_percentile,
                upper_percentile,
            ),
            patch_artist=True,
            showfliers=outlier,
            boxprops={
                "facecolor": color,
                "alpha": 0.8,
                "edgecolor": color,
                "linewidth": 0.7,
            },
            whiskerprops={
                "color": color,
                "alpha": 1,
                "linewidth": 1,
                "linestyle": "--",
            },
            capprops={
                "color": color,
                "alpha": 1,
                "linewidth": 2,
                "linestyle": "--",
            },
            medianprops={"color": "black", "linewidth": 1, "alpha": 1},
            flierprops={
                "marker": "o",
                "alpha": 1,
                "markeredgewidth": 0.3 + (0.4 * (1 - box_width)),
                "markeredgecolor": color,
                "markerfacecolor": "none",
            },
        )
    if mean:
        means = np.array([np.nanmean(arr) for arr in data], dtype=float)
        axes.plot(
            positions,
            means,
            "D",
            markersize=4,
            color="black",
            zorder=3,  # Above boxes and scatter
        )
    if scatter:
        rng = np.random.default_rng(42)
        jitter = box_width * 0.5

        x_points: list[np.ndarray] = []
        y_points: list[np.ndarray] = []
        for position, data_points in zip(positions, data, strict=True):
            x_points.append(
                position + rng.uniform(-jitter / 2, jitter / 2, size=len(data_points))
            )
            y_points.append(data_points)

        x_all = np.concatenate(x_points)
        y_all = np.concatenate(y_points)

        axes.scatter(
            x_all,
            y_all,
            color=color,
            alpha=0.35,
            linewidths=0,
            zorder=2,  # above bands/boxes
        )

    config.add_legend_item(
        legend_label,
        Line2D(
            [],
            [],
            marker="s",
            linestyle="None",
            color=color,
            label=legend_label,
        ),
    )


def generate_legend_items(
    plot_context: PlotContext, lower_percentile: int, upper_percentile: int
) -> None:
    config = plot_context.plotConfig()
    outlier = plot_context.outliers
    scatter = plot_context.scatter_plot
    box = plot_context.box_plot
    mean = plot_context.mean
    if box:
        config.add_legend_item(
            "Median", Line2D([0], [0], color="black", linewidth=0.9, alpha=1)
        )
        config.add_legend_item(
            (f"Whiskers ({lower_percentile}-{upper_percentile} %)"),
            Line2D([0], [0], color="black", linewidth=2, linestyle="--", alpha=1),
        )
        if outlier:
            config.add_legend_item(
                "Outliers",
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markeredgecolor="black",
                    markerfacecolor="none",
                    markersize=6,
                    alpha=1,
                ),
            )
    if scatter:
        config.add_legend_item(
            "Scatter points",
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markeredgecolor="None",
                linestyle="None",
                alpha=0.35,
            ),
        )
    if mean:
        config.add_legend_item(
            "Mean",
            Line2D(
                [0],
                [0],
                marker="D",
                color="black",
                markersize=4,
                linestyle="None",
                alpha=1,
            ),
        )
