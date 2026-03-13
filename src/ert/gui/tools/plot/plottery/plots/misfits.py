from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, cast

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
        if response_type in {"summary", "breakthrough"}:
            key_index_with_correct_dtype = pl.col("key_index").str.to_datetime(
                strict=False
            )
        elif response_type in {"gen_data", "rft"}:
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

        self._misfit_boxplot(
            figure,
            data_with_misfits,
            plot_context,
        )

    def _misfit_boxplot(
        self,
        figure: Figure,
        data_with_misfits: dict[tuple[str, str], pl.DataFrame],
        plot_context: PlotContext,
    ) -> None:
        ensemble_to_misfit_df = {
            k: v for k, v in data_with_misfits.items() if not v.is_empty()
        }
        sorted_ensemble_keys = sorted(ensemble_to_misfit_df.keys())

        all_misfits = pl.concat(
            [
                df.with_columns(pl.lit(key).alias("ensemble_key"))
                for key, df in ensemble_to_misfit_df.items()
            ]
        ).select(["Realization", "key_index", "misfit", "ensemble_key"])

        distinct_gendata_index = all_misfits["key_index"].unique().sort().to_list()

        color_map = self._make_ensemble_colors(sorted_ensemble_keys)
        self._draw_legend(
            figure=figure,
            ensemble_colors=color_map,
            sorted_ensemble_keys=sorted_ensemble_keys,
        )

        num_cols = len(distinct_gendata_index)
        axes = figure.subplots(nrows=2, ncols=num_cols, sharex="col", sharey=True)
        axes = (
            axes.reshape(2, num_cols)
            if num_cols > 1
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

                num_points = len(mis_vals)

                if num_points >= 200:
                    marker_size = 2
                elif num_points >= 100:
                    marker_size = 3
                else:
                    marker_size = 4

                sns.stripplot(
                    x=[x_center] * num_points,  # Plot all points at the same x-center
                    y=mis_vals,
                    ax=ax_bottom,
                    color=color,
                    size=marker_size,
                    alpha=0.35,
                    jitter=True,  # Spread points sharing same x-center horizontally
                )

        y_min, y_max = self._compute_misfits_padded_minmax(all_misfits, 0.05)
        self._style_boxplots(axes, x_positions, y_max, y_min)

        for ax, key_val in zip(axes_bottom, distinct_gendata_index, strict=True):
            label = (
                key_val.date()
                if isinstance(key_val, datetime)
                else f"index={int(key_val)}"
            )
            ax.set_xlabel(label, rotation=25, ha="right")

        figure.suptitle(
            f"{plot_context.key()} (Signed Chi-squared misfits)",
            fontsize=14,
            y=0.98,
        )
        figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.88))

    def _style_boxplots(
        self,
        axes: Any | np.ndarray[tuple[Any, ...], np.dtype[Any]],
        x_positions: np.ndarray[tuple[int]],
        y_max: float,
        y_min: float,
    ) -> None:
        """Styles the frames containing individual misfit plots to create a prettier
        collection of plots."""
        (n_rows, n_cols) = axes.shape
        for r_idx in range(n_rows):
            for c_idx in range(n_cols):
                ax = axes[r_idx, c_idx]
                self._fade_axis_ticklabels(ax)

                # Style horizontal dotted lines
                ax.set(ylim=(y_min, y_max))
                ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
                ax.grid(True, axis="y", linestyle=":", alpha=0.4)

                is_first_col = c_idx == 0
                is_bottom_row = r_idx == (n_rows - 1)

                # Visualize only left-most and bottom-most borders
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(is_first_col)
                ax.spines["bottom"].set_visible(is_bottom_row)

                # Disable ticks on other than left-most and bottom-most borders
                ax.set_xticks(x_positions, labels=[])
                if not is_bottom_row:
                    ax.tick_params(axis="x", which="both", bottom=False)
                if not is_first_col:
                    ax.tick_params(axis="y", which="both", left=False)
