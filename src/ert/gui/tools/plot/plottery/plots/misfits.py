from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.figure import Figure

    from ert.gui.tools.plot.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.tools.plot.plottery import PlotContext


class MisfitsPlot:
    """
    Visualize signed chi-squared misfits between simulated responses and observations.

    Layout:
        - X-axis: index for gen data, time for summary
        - Y-axis: signed chi-squared misfits
        - Glyphs: One boxplot per time step / gendata index
    """

    def __init__(self) -> None:
        self.dimensionality = 2
        self.requires_observations = True

    @staticmethod
    def _fade_axis_ticklabels(ax: plt.Axes) -> None:
        """Apply the same alpha to all tick labels on an axis."""
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_alpha(0.4)

        ax.xaxis.label.set_alpha(0.4)
        ax.yaxis.label.set_alpha(0.4)

    @staticmethod
    def _draw_legend(
        figure: Figure,
        ensemble_colors: dict[tuple[str, str], str],
        sorted_ensemble_keys: list[tuple[str, str]],
    ) -> None:
        legend_handles = [
            Line2D(
                [],
                [],
                marker="s",
                linestyle="None",
                color=ensemble_colors[key],
                label=key[0],
            )
            for key in sorted_ensemble_keys
        ]
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=min(len(legend_handles), 4),
        )

    @staticmethod
    def _make_ensemble_colors(
        sorted_ensemble_keys: list[tuple[str, str]],
    ) -> dict[tuple[str, str], str]:
        """
        Build a consistent color map and figure-level legend
        used by all misfit plots
        """
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return {
            key: colors[i % len(colors)] for i, key in enumerate(sorted_ensemble_keys)
        }

    @staticmethod
    def _compute_misfits_padded_minmax(
        misfits_df: pl.DataFrame, relative_pad_y_axis: float
    ) -> tuple[float, float]:
        y_min, y_max = (
            cast(float, misfits_df["misfit"].min()),
            cast(float, misfits_df["misfit"].max()),
        )
        abs_pad_y = (y_max - y_min) * relative_pad_y_axis
        y_min -= abs_pad_y
        y_max += abs_pad_y

        return y_min, y_max

    @staticmethod
    def _wide_pandas_to_long_polars_with_misfits(
        ensemble_to_data_map: dict[tuple[str, str], pd.DataFrame],
        observation_data: pd.DataFrame,
        response_type: Literal["summary", "gen_data"],
    ) -> dict[tuple[str, str], pl.DataFrame]:
        if response_type == "summary":
            key_index_with_correct_dtype = pl.col("key_index").str.to_datetime(
                strict=False
            )
        elif response_type == "gen_data":
            key_index_with_correct_dtype = (
                pl.col("key_index").cast(pl.Float32).cast(pl.UInt16)
            )
        else:
            raise ValueError(f"Unsupported response_type: {response_type}")

        obs_df = (
            pl.from_pandas(observation_data.T)
            .rename({"OBS": "observation", "STD": "error"})
            .with_columns(pl.col("key_index").cast(pl.String))
            .with_columns(key_index_with_correct_dtype)
        )

        return {
            ens_key: (
                pl.from_pandas(df, include_index=True)
                .unpivot(
                    index=df.index.name,
                    variable_name="key_index",
                    value_name="response",
                )
                .with_columns(key_index_with_correct_dtype)
                .join(obs_df, on="key_index", how="inner")
                .with_columns(
                    (pl.col("response") - pl.col("observation")).alias("residual")
                )
                .with_columns(
                    (
                        pl.col("residual").sign()
                        * (pl.col("residual") / pl.col("error")).pow(2)
                    ).alias("misfit")
                )
                .drop("residual")
            )
            for ens_key, df in ensemble_to_data_map.items()
        }

    def plot(
        self,
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_images: dict[str, npt.NDArray[np.float32]],
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None:
        assert key_def is not None
        response_type = key_def.metadata["data_origin"]
        data_with_misfits = self._wide_pandas_to_long_polars_with_misfits(
            {(eo.name, eo.id): df for eo, df in ensemble_to_data_map.items()},
            observation_data,
            response_type,
        )

        if response_type == "summary":
            self._plot_summary_misfits_boxplots(
                figure,
                data_with_misfits,
                plot_context,
            )

        elif response_type == "gen_data":
            self._plot_gendata_misfits(
                figure,
                data_with_misfits,
                plot_context,
            )

    def _plot_gendata_misfits(
        self,
        figure: Figure,
        data_with_misfits: dict[tuple[str, str], pl.DataFrame],
        plot_context: PlotContext,
    ) -> None:
        # Only plot ensembles with data (i.e., they pertain to an experiment
        # with observations, and there are responses towards those in the ens)
        ensemble_to_misfit_df = {
            k: v for k, v in data_with_misfits.items() if not v.is_empty()
        }
        sorted_ensemble_keys = sorted(ensemble_to_misfit_df.keys())

        all_misfits = pl.concat(
            [
                df.with_columns(pl.lit(key).alias("ensemble_key"))
                for key, df in ensemble_to_misfit_df.items()
            ]
        ).select(["realization", "key_index", "misfit", "ensemble_key"])

        distinct_gendata_index = all_misfits["key_index"].unique().sort().to_list()
        num_gendata_index = len(distinct_gendata_index)

        color_map = self._make_ensemble_colors(sorted_ensemble_keys)
        self._draw_legend(
            figure=figure,
            ensemble_colors=color_map,
            sorted_ensemble_keys=sorted_ensemble_keys,
        )

        y_min, y_max = self._compute_misfits_padded_minmax(all_misfits, 0.05)

        # Create subplot grid (2 rows, N columns)
        axes = figure.subplots(
            nrows=2, ncols=num_gendata_index, sharex="col", sharey=True
        )
        axes = (
            axes.reshape(2, num_gendata_index)
            if num_gendata_index > 1
            else np.array([[axes[0]], [axes[1]]])
        )
        axes_top, axes_bottom = axes[0, :], axes[1, :]

        x_positions = np.arange(len(sorted_ensemble_keys))
        box_width_relative = 0.6

        for col_idx, key_index in enumerate(distinct_gendata_index):
            ax_top, ax_bottom = axes_top[col_idx], axes_bottom[col_idx]

            for ens_idx, ens_key in enumerate(sorted_ensemble_keys):
                color = color_map.get(ens_key, "C0")

                # Filter for the specific key and ensemble
                mis_vals = all_misfits.filter(
                    (pl.col("key_index") == key_index)
                    & (pl.col("ensemble_key") == ens_key)
                )["misfit"].to_numpy()

                if mis_vals.size == 0:
                    continue

                x_center = x_positions[ens_idx]

                # Top: Boxplot
                ax_top.boxplot(
                    mis_vals,
                    positions=[x_center],
                    widths=box_width_relative,
                    patch_artist=True,
                    showfliers=False,
                    boxprops={"facecolor": color, "alpha": 0.35},
                    whiskerprops={"color": color, "alpha": 0.8},
                    capprops={"color": color, "alpha": 0.8},
                    medianprops={"color": color, "alpha": 0.8},
                )
                ax_top.plot(x_center, np.mean(mis_vals), "o", markersize=4, color=color)

                # Bottom: Strip plot with dynamic marker size
                num_points = len(mis_vals)

                if num_points >= 200:
                    marker_size = 2
                elif num_points >= 100:
                    marker_size = 3
                else:
                    marker_size = 4

                # Use stripplot as a robust alternative for dense data
                sns.stripplot(
                    x=[x_center] * num_points,  # Plot all points at the same x-center
                    y=mis_vals,
                    ax=ax_bottom,
                    color=color,
                    size=marker_size,
                    alpha=0.35,
                    jitter=True,  # Explicitly add jitter
                )

        # Axis/spine styling
        (n_rows, n_cols) = axes.shape
        for r_idx in range(n_rows):
            for c_idx in range(n_cols):
                ax = axes[r_idx, c_idx]
                is_first_col = c_idx == 0
                is_bottom_row = r_idx == (n_rows - 1)

                # Common styling
                ax.set(ylim=(y_min, y_max))
                ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
                ax.grid(True, axis="y", linestyle=":", alpha=0.4)
                self._fade_axis_ticklabels(ax)

                # Spines
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(is_first_col)
                ax.spines["bottom"].set_visible(is_bottom_row)

                # Ticks
                ax.set_xticks(x_positions, labels=[])
                if not is_bottom_row:
                    ax.tick_params(axis="x", which="both", bottom=False)
                if not is_first_col:
                    ax.tick_params(axis="y", which="both", left=False)

        for ax, key_val in zip(axes_bottom, distinct_gendata_index, strict=True):
            ax.set_xlabel(f"index={int(key_val)}", rotation=25, ha="right")

        figure.suptitle(
            f"{plot_context.key()} (Signed Chi-squared misfits per index)",
            fontsize=14,
            y=0.98,
        )
        figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.88))

    def _plot_summary_misfits_boxplots(
        self,
        figure: Figure,
        data_with_misfits: dict[tuple[str, str], pl.DataFrame],
        plot_context: PlotContext,
    ) -> None:
        # Calculate shared y-axis limits from all misfits
        all_misfits = pl.concat(
            [df.select("misfit") for df in data_with_misfits.values()]
        )
        y_min, y_max = self._compute_misfits_padded_minmax(all_misfits, 0.05)

        # Prepare ensemble colors and draw the legend
        sorted_ensemble_keys = sorted(data_with_misfits.keys())
        color_map = self._make_ensemble_colors(sorted_ensemble_keys)
        self._draw_legend(
            figure=figure,
            ensemble_colors=color_map,
            sorted_ensemble_keys=sorted_ensemble_keys,
        )

        # Create all subplots at once with shared axes
        n_ens = len(sorted_ensemble_keys)
        axes = figure.subplots(nrows=n_ens, ncols=1, sharex=True, sharey=True)
        axes = [axes] if n_ens == 1 else axes.tolist()
        axes[0].set_ylim(y_min, y_max)

        for ax, ensemble_key in zip(axes, sorted_ensemble_keys, strict=True):
            df = data_with_misfits[ensemble_key]
            if df.is_empty():
                continue

            df = df.select(["key_index", "misfit"]).sort("key_index")
            times_py = df["key_index"].unique(maintain_order=True).to_list()
            positions = mdates.date2num(times_py)  # type: ignore[no-untyped-call]

            # Calculate dynamic box width based on time spacing
            min_dt = np.min(np.diff(positions)) if len(positions) > 1 else 1.0

            # multiplier to downsize outlier sizes etc
            # (without this, outliers, whiskers etc are sized way
            # out of proportion when there are many tiny boxplots)
            many_boxes_factor = min(1, len(times_py) / 50)
            box_width = min_dt * (0.7 - 0.3 * (1 - many_boxes_factor))

            # One boxplot per time step
            grouped_misfits = df.group_by("key_index", maintain_order=True).agg(
                pl.col("misfit")
            )
            data_for_boxes = [
                s.to_numpy() if s.len() > 0 else np.array([np.nan])
                for s in grouped_misfits["misfit"]
            ]

            color = color_map.get(ensemble_key)

            # Draw the boxplots with inlined styles
            bp = ax.boxplot(
                data_for_boxes,
                positions=positions,
                widths=box_width,
                whis=(5, 95),
                showfliers=True,
                manage_ticks=False,
                patch_artist=True,
                boxprops={
                    "facecolor": color,
                    "alpha": 0.18,
                    "edgecolor": color,
                    "linewidth": 0.7,
                },
                whiskerprops={"color": color, "alpha": 0.6, "linewidth": 0.7},
                capprops={"color": color, "alpha": 0.6, "linewidth": 0.7},
                medianprops={"color": color, "linewidth": 0.6, "alpha": 0.9},
                flierprops={
                    "marker": "o",
                    "markersize": min(6, box_width * (0.4 - (0.2 * many_boxes_factor))),
                    "alpha": 0.7,
                    "markeredgewidth": 0.3 + (0.4 * (1 - many_boxes_factor)),
                    "markeredgecolor": color,
                    "markerfacecolor": "none",
                },
            )
            plt.setp(bp["fliers"], zorder=1.5)  # Put fliers behind other elements

            # Add ensemble name text and a horizontal line at y=0
            ax.text(
                0.01,
                0.85,
                f"{ensemble_key[0]}",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=9,
                alpha=0.75,
            )
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4, zorder=0)

        # Apply common styling to all axes
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(True)
            ax.tick_params(axis="y", labelsize=10, width=0.5, length=3)
            ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.75)
            self._fade_axis_ticklabels(ax)

        # Hide the x-axis on all but the last plot
        for ax in axes[:-1]:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # Style the x-axis only on the last plot
        bottom_ax = axes[-1]
        bottom_ax.spines["bottom"].set_visible(True)
        bottom_ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # type: ignore[no-untyped-call]
        bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # type: ignore[no-untyped-call]

        bottom_ax.tick_params(axis="x", labelsize=8, width=0.5, length=3)
        plt.setp(bottom_ax.get_xticklabels(), rotation=25, ha="right")

        figure.suptitle(
            f"{plot_context.key()} (Signed Chi-squared misfits over time)",
            fontsize=14,
            y=0.97,
            alpha=0.9,
        )
        figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
