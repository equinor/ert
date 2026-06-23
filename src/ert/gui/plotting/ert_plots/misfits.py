from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.lines import Line2D

from ert.gui.plotting.utils import PlotTools
from ert.gui.plotting.utils.plot_context import PlotType

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.figure import Figure

    from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.plotting.utils import PlotContext
    from ert.gui.plotting.utils.plot_types import ObservationPlotLocations

UPPER_PERCENTILE_FOR_WHISKERS = 95
LOWER_PERCENTILE_FOR_WHISKERS = 5


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
    def _show_no_data(figure: Figure, message: str) -> None:
        axes = figure.add_subplot(111)
        axes.text(0.5, 0.5, message, ha="center", va="center")
        axes.set_axis_off()

    @staticmethod
    def _map_ensembles_to_colours(
        sorted_ensemble_keys: list[tuple[str, str]],
        colour_list: list[tuple[str, float]],  # tuples of hex and alpha values
    ) -> dict[tuple[str, str], str]:
        """
        Build a consistent color map and figure-level legend
        used by all misfit plots. Alpha values are dropped.
        """
        return {
            key: colour_list[i % len(colour_list)][0]
            for i, key in enumerate(sorted_ensemble_keys)
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
        obs_loc: ObservationPlotLocations | None,
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None:
        assert key_def is not None
        if observation_data.empty:
            self._show_no_data(figure, "No observations available")
            return

        response_type = key_def.metadata["data_origin"]
        data_with_misfits = self._wide_pandas_to_long_polars_with_misfits(
            {(eo.name, eo.id): df for eo, df in ensemble_to_data_map.items()},
            observation_data,
            response_type,
        )

        if all(df.is_empty() for df in data_with_misfits.values()):
            self._show_no_data(figure, "No misfit data available")
            return

        if response_type in {"summary", "breakthrough"}:
            self._plot_misfits(
                figure,
                data_with_misfits,
                plot_context,
            )

        elif response_type in {"gen_data", "rft"}:
            self._plot_misfits(
                figure,
                data_with_misfits,
                plot_context,
                summary_or_breakthrough=False,
            )

    def _plot_misfits(
        self,
        figure: Figure,
        data_with_misfits: dict[tuple[str, str], pl.DataFrame],
        plot_context: PlotContext,
        *,
        summary_or_breakthrough: bool = True,
    ) -> None:

        all_misfits = pl.concat(
            [df.select(["misfit", "key_index"]) for df in data_with_misfits.values()]
        )

        all_unique_indexes = all_misfits["key_index"].unique().sort().to_list()
        plot_context.plot_type = PlotType.BOX
        config = plot_context.plotConfig()
        outlier = plot_context.outliers
        scatter = plot_context.scatter_plot
        box = plot_context.box_plot
        mean = plot_context.mean

        index_to_pos = {idx: i for i, idx in enumerate(all_unique_indexes)}

        many_boxes_factor = min(1, len(all_unique_indexes) / 50)
        sorted_ensemble_keys = sorted(data_with_misfits.keys())
        color_map = self._map_ensembles_to_colours(
            sorted_ensemble_keys, plot_context.plotConfig().line_color_cycle()
        )

        y_min, y_max = self._compute_misfits_padded_minmax(all_misfits, 0.05)

        axes = figure.add_subplot(111)
        axes.set_ylim(y_min, y_max)

        for idx in index_to_pos.values():
            if idx % 2 == 0:
                axes.axvspan(idx - 0.5, idx + 0.5, color="grey", alpha=0.07, zorder=0)

        n_ens = len(sorted_ensemble_keys)
        box_width = 0.8 / n_ens

        axes.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)

        for ens_idx, ens_key in enumerate(sorted_ensemble_keys):
            color = color_map.get(ens_key)

            df = data_with_misfits[ens_key]
            if df.is_empty():
                continue

            df = df.select(["key_index", "misfit"]).sort("key_index")
            grouped_misfits = df.group_by("key_index", maintain_order=True).agg(
                pl.col("misfit")
            )

            offset = (ens_idx - (n_ens - 1) / 2) * box_width
            positions = [
                index_to_pos[idx] + offset
                for idx in grouped_misfits["key_index"].to_list()
            ]
            data_for_boxes = [
                s.to_numpy() if s.len() > 0 else np.array([np.nan])
                for s in grouped_misfits["misfit"]
            ]
            if box:
                axes.boxplot(
                    data_for_boxes,
                    positions=positions,
                    widths=box_width,
                    whis=(LOWER_PERCENTILE_FOR_WHISKERS, UPPER_PERCENTILE_FOR_WHISKERS),
                    showfliers=outlier,
                    manage_ticks=False,
                    patch_artist=True,
                    boxprops={
                        "facecolor": color,
                        "alpha": 0.8,
                        "edgecolor": color,
                        "linewidth": 0.7,
                    },
                    whiskerprops={
                        "color": color,
                        "alpha": 1,
                        "linewidth": 0.8,
                        "linestyle": "--",
                    },
                    capprops={"color": color, "alpha": 1, "linewidth": 0.8},
                    medianprops={"color": "black", "linewidth": 0.8, "alpha": 1},
                    flierprops={
                        "marker": "o",
                        "alpha": 1,
                        "markeredgewidth": 0.3 + (0.4 * (1 - many_boxes_factor)),
                        "markeredgecolor": color,
                        "markerfacecolor": "none",
                    },
                )

            if mean:
                means = np.array(
                    [np.nanmean(arr) for arr in data_for_boxes], dtype=float
                )
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
                for position, box_data in zip(positions, data_for_boxes, strict=True):
                    x_points.append(
                        position
                        + rng.uniform(-jitter / 2, jitter / 2, size=len(box_data))
                    )
                    y_points.append(box_data)

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
                ens_key[0],
                Line2D(
                    [],
                    [],
                    marker="s",
                    linestyle="None",
                    color=color,
                    label=ens_key[0],
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

        if box:
            config.add_legend_item(
                "Median", Line2D([0], [0], color="black", linewidth=0.6, alpha=1)
            )
            config.add_legend_item(
                "Whiskers (5-95%)",
                Line2D([0], [0], color="black", linewidth=0.7, linestyle="--", alpha=1),
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
        if outlier and box:
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

        axes.set_xlim(-0.5, len(all_unique_indexes) - 0.5)
        if summary_or_breakthrough:
            axes.set_xticks(
                list(index_to_pos.values()),
                labels=[ts.strftime("%Y-%m-%d") for ts in all_unique_indexes],
            )
        else:
            axes.set_xticks(
                np.arange(len(all_unique_indexes)),
                labels=[str(int(k)) for k in all_unique_indexes],
            )

        index_string = "timestep" if summary_or_breakthrough else "index"
        plot_context.plotConfig().set_title(
            f"{plot_context.key()} (Signed Chi-squared misfits per {index_string})"
        )
        PlotTools.finalizePlot(
            plot_context,
            figure,
            axes,
            default_x_label="",
            default_y_label="Value",
        )
