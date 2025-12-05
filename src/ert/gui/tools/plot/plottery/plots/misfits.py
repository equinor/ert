from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl

from .plot_tools import PlotTools  # Utility functions for plot finalization

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.figure import Figure

    from ert.gui.tools.plot.plot_api import EnsembleObject
    from ert.gui.tools.plot.plottery import PlotContext


class MisfitsPlot:
    """
    Visualize signed misfits between simulated responses and observations.

    Layout:
        - 2 x N grid of subplots, where N is number of observation indices.
        - Each column corresponds to one observation:
            * Top: boxplots of misfits per ensemble.
            * Bottom: jittered scatter of misfits per ensemble.
        - All subplots share the same y-axis range.
    """

    def __init__(self) -> None:
        self.dimensionality = 2
        self.requires_observations = True

    def plot(
        self,
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_images: dict[str, npt.NDArray[np.float32]],
    ) -> None:
        import numpy as np
        import polars as pl
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D

        config = plot_context.plotConfig()
        misfit_plot_mode = config.getMisfitsPlotMode()

        # ------------------------------------------------------------------
        # Observation metadata: expect columns "STD", "OBS", "key_index"
        # ------------------------------------------------------------------
        observation_data_columnar = (
            pl.from_pandas(observation_data.T)
            .with_columns(pl.col("key_index").cast(pl.String))
            .rename(
                {
                    "OBS": "observations",
                    "STD": "errors",
                }
            )
        )

        response_data_columnar = {
            k: pl.from_pandas(v) for k, v in ensemble_to_data_map.items()
        }

        if observation_data_columnar.is_empty():
            ax = figure.add_subplot(1, 1, 1)
            ax.text(
                0.5,
                0.5,
                "No observations available for this key.",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            PlotTools.finalizePlot(
                plot_context,
                figure,
                ax,
                default_x_label="",
                default_y_label="",
            )
            return

        obs = observation_data_columnar.with_columns(
            pl.col("key_index").cast(pl.Float64)
        )

        keys = (
            obs.select("key_index")
            .unique()
            .sort("key_index")
            .get_column("key_index")
            .to_list()
        )

        if not keys:
            ax = figure.add_subplot(1, 1, 1)
            ax.text(
                0.5,
                0.5,
                "No matching observation indices.",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            return

        # Try to detect a time-like column for summary mode labelling.
        time_col = None
        if misfit_plot_mode != "Univariate":
            for c in observation_data_columnar.columns:
                lc = c.lower()
                if lc in ("time", "date", "datetime", "timestep", "obs_time"):
                    time_col = c
                    break

        # ------------------------------------------------------------------
        # Collect misfit ensembles (name endswith ".misfits")
        # ------------------------------------------------------------------
        misfits_long_list: list[pl.DataFrame] = []
        base_to_misfit_name: dict[str, str] = {}

        for eo, df in response_data_columnar.items():
            if not eo.name.endswith(".misfits"):
                continue

            misfits_wide = df
            if "Unnamed: 0" in misfits_wide.columns:
                misfits_wide = misfits_wide.drop("Unnamed: 0")

            base_name = eo.name.removesuffix(".misfits")

            misfits_long_eo = (
                misfits_wide.with_row_count("realization")
                .melt(
                    id_vars=["realization"],
                    variable_name="key_index",
                    value_name="misfit",
                )
                .with_columns(
                    pl.col("key_index").cast(pl.Float64),
                    pl.col("misfit"),  # signed misfit
                    pl.lit(eo.name).alias("ensemble_name"),
                    pl.lit(base_name).alias("ensemble_base"),
                )
                .filter(pl.col("key_index").is_in(keys))
            )

            misfits_long_list.append(misfits_long_eo)
            base_to_misfit_name[base_name] = eo.name

        misfits_long = (
            pl.concat(misfits_long_list, how="vertical") if misfits_long_list else None
        )

        if misfits_long is None or misfits_long.height == 0:
            ax = figure.add_subplot(1, 1, 1)
            ax.text(
                0.5,
                0.5,
                "No misfit data available.",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            return

        # ------------------------------------------------------------------
        # Shared base-name ordering & colour map
        # ------------------------------------------------------------------
        base_names = sorted(base_to_misfit_name.keys())
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colour_map = {
            base: colours[i % len(colours)] for i, base in enumerate(base_names)
        }

        # Global symmetric y-limits across ALL axes
        global_min = float(misfits_long["misfit"].min())
        global_max = float(misfits_long["misfit"].max())
        max_abs = max(abs(global_min), abs(global_max)) or 1.0
        y_min = -1.05 * max_abs
        y_max = 1.05 * max_abs

        # ------------------------------------------------------------------
        # Create small-multiples layout: 2 rows, len(keys) columns
        # Share y across the whole grid for interactive sync
        # ------------------------------------------------------------------
        n_keys = len(keys)
        axes_top: list = []
        axes_bottom: list = []

        ref_axis = None

        for i in range(n_keys):
            if i == 0:
                ax_top = figure.add_subplot(2, n_keys, 1)
                ax_bottom = figure.add_subplot(
                    2, n_keys, n_keys + 1, sharex=ax_top, sharey=ax_top
                )
                ref_axis = ax_top
            else:
                ax_top = figure.add_subplot(2, n_keys, i + 1, sharey=ref_axis)
                ax_bottom = figure.add_subplot(
                    2,
                    n_keys,
                    n_keys + i + 1,
                    sharex=axes_top[0],
                    sharey=ref_axis,
                )

            axes_top.append(ax_top)
            axes_bottom.append(ax_bottom)

        # ------------------------------------------------------------------
        # TOP ROW: boxplots per ensemble, per observation index
        # ------------------------------------------------------------------
        legend_handles: list[Line2D] = []

        if base_names:
            x_positions_local = np.arange(len(base_names))
            box_width = 0.6
        else:
            x_positions_local = np.array([0.0])
            box_width = 0.5

        for col_idx, key in enumerate(keys):
            ax = axes_top[col_idx]

            for i, base in enumerate(base_names):
                colour = colour_map.get(base, "C0")
                ens_name = base_to_misfit_name[base]

                sub = misfits_long.filter(
                    (pl.col("ensemble_base") == base) & (pl.col("key_index") == key)
                )
                mis_vals = sub["misfit"].to_numpy()
                if mis_vals.size == 0:
                    continue

                x_center = i
                pos = [x_center]

                bp = ax.boxplot(
                    mis_vals,
                    positions=pos,
                    vert=True,
                    widths=box_width,
                    patch_artist=True,
                    manage_ticks=False,
                    showfliers=False,
                )

                for box in bp["boxes"]:
                    box.set_facecolor(colour)
                    box.set_alpha(0.35)

                for element in ["whiskers", "caps", "medians"]:
                    for artist in bp[element]:
                        artist.set_color(colour)
                        artist.set_alpha(0.8)

                mean_misfit = float(np.mean(mis_vals))
                ax.plot(
                    [x_center],
                    [mean_misfit],
                    marker="o",
                    markersize=4,
                    color=colour,
                )

                # Build legend handles from first column only
                if col_idx == 0:
                    legend_handles.append(
                        Line2D(
                            [],
                            [],
                            marker="s",
                            linestyle="None",
                            color=colour,
                            label=ens_name,
                        )
                    )

            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, axis="y", linestyle=":", alpha=0.4)
            ax.set_xticks(x_positions_local)
            ax.set_xticklabels([])  # no category labels, only bottom xlabel

        # ------------------------------------------------------------------
        # BOTTOM ROW: jittered scatter per ensemble, per observation index
        # ------------------------------------------------------------------
        jitter_halfwidth = 0.35 if base_names else 0.0

        for col_idx, key in enumerate(keys):
            ax = axes_bottom[col_idx]

            for i, base in enumerate(base_names):
                colour = colour_map.get(base, "C0")

                sub = misfits_long.filter(
                    (pl.col("ensemble_base") == base) & (pl.col("key_index") == key)
                )
                vals = sub["misfit"].to_numpy()
                if vals.size == 0:
                    continue

                x_center = i
                if jitter_halfwidth > 0:
                    jitter = (np.random.rand(len(vals)) - 0.5) * 2.0 * jitter_halfwidth
                else:
                    jitter = np.zeros_like(vals)

                ax.scatter(
                    x_center + jitter,
                    vals,
                    s=8,
                    alpha=0.6,
                    color=colour,
                )

            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, axis="y", linestyle=":", alpha=0.4)
            ax.set_xticks(x_positions_local)
            ax.set_xticklabels([])

        # ------------------------------------------------------------------
        # Legend (above all plots, not covering any subplot)
        # ------------------------------------------------------------------
        if legend_handles:
            figure.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.93),  # LOWER than before
                ncol=min(len(legend_handles), 4),
            )

        # ------------------------------------------------------------------
        # Let PlotTools handle main styling / title, then move title to suptitle
        # ------------------------------------------------------------------
        # PlotTools.finalizePlot(
        #     plot_context,
        #     figure,
        #     axes_bottom[0],
        # )

        # Use whatever title finalizePlot set on the reference axis as a figure title.
        ref_title = axes_bottom[0].get_title()
        if ref_title:
            figure.suptitle(ref_title)
            for ax in (*axes_top, *axes_bottom):
                ax.set_title("")

        # ------------------------------------------------------------------
        # Styling: ticks, spines, y-labels
        #   - No vertical tick marks.
        #   - Only outer spines visible (no spines between charts).
        #   - y-label only on bottom-left subplot.
        # ------------------------------------------------------------------
        for row_idx, (ax_top, ax_bottom) in enumerate(zip(axes_top, axes_bottom)):
            # No vertical tick marks on any x-axis
            ax_top.tick_params(axis="x", which="both", length=0)
            ax_bottom.tick_params(axis="x", which="both", length=0)

        # --- Spine logic: clean minimal outer frame ---
        for col_idx, (ax_top, ax_bottom) in enumerate(zip(axes_top, axes_bottom)):
            # 1) Hide ALL spines on both axes first
            for ax in (ax_top, ax_bottom):
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["bottom"].set_visible(False)

            # 2) Show left spine only on the leftmost column
            if col_idx == 0:
                ax_top.spines["left"].set_visible(True)
                ax_bottom.spines["left"].set_visible(True)

            # 3) Show bottom spine only on the bottom row (ax_bottom)
            ax_bottom.spines["bottom"].set_visible(True)

        # y-axis labels: only bottom-left
        for ax in axes_top:
            ax.set_ylabel("")
        for ax in axes_bottom:
            ax.set_ylabel("")
        if axes_bottom:
            axes_bottom[0].set_ylabel("Misfit")

        # --- Y tick + label visibility ---
        for col_idx, (ax_top, ax_bottom) in enumerate(zip(axes_top, axes_bottom)):
            if col_idx == 0:
                # LEFTMOST COLUMN → show ticks + labels
                ax_top.tick_params(
                    axis="y",
                    which="both",
                    left=True,
                    right=False,
                    labelleft=True,
                    labelright=False,
                )
                ax_bottom.tick_params(
                    axis="y",
                    which="both",
                    left=True,
                    right=False,
                    labelleft=True,
                    labelright=False,
                )
            else:
                # ALL OTHER COLUMNS → hide ticks + labels entirely
                ax_top.tick_params(
                    axis="y",
                    which="both",
                    left=False,
                    right=False,
                    labelleft=False,
                    labelright=False,
                )
                ax_bottom.tick_params(
                    axis="y",
                    which="both",
                    left=False,
                    right=False,
                    labelleft=False,
                    labelright=False,
                )

        axes_top[0].set_ylabel("χ²")
        axes_bottom[0].set_ylabel("χ²")
        # ------------------------------------------------------------------
        # Column labels at the bottom: "index=..." or "time=..."
        # ------------------------------------------------------------------
        for col_idx, key in enumerate(keys):
            ax = axes_bottom[col_idx]

            if misfit_plot_mode == "Univariate":
                label = f"index={int(key)}"
            else:
                label = None
                if time_col is not None:
                    sub_obs = obs.filter(pl.col("key_index") == key)
                    if sub_obs.height > 0:
                        t_val = sub_obs.select(time_col).to_series()[0]
                        label = f"time={t_val}"
                if label is None:
                    label = f"time={key}"

            ax.set_xlabel(label)

        for ax in axes_top + axes_bottom:
            # Set opacity of all tick labels
            for label in ax.get_yticklabels():
                label.set_alpha(0.7)
            for label in ax.get_xticklabels():
                label.set_alpha(0.7)

        # Final layout, leaving room for legend and suptitle
        figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.88))
        figure.suptitle(
            plot_context.key() + " (Signed Chi-squared)", fontsize=14, y=0.98
        )
