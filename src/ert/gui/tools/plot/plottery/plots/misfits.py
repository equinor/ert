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
    A plotter for visualizing misfits between simulated responses and observations.
    Supports 'Univariate' (boxplots over time/index) and 'Summary' (histogram of all misfits) modes.
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
        axes_misfit = figure.add_subplot(2, 1, 1)
        axes_resp = figure.add_subplot(2, 1, 2, sharex=axes_misfit)
        response_key = plot_context.key()
        misfit_plot_mode = config.getMisfitsPlotMode()

        # columns: "STD", "OBS", "key_index"
        observation_data_columnar = (
            pl.from_pandas(observation_data.T)
            .with_columns(pl.col("key_index").cast(pl.String))
            .rename(
                {
                    "OBS":"observations",
                    "STD":"errors",
                }
            )
        )

        response_data_columnar = {
            k:pl.from_pandas(v) for k, v in ensemble_to_data_map.items()
        }

        if observation_data_columnar.is_empty():
            for ax in (axes_misfit, axes_resp):
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
                plot_context, figure, axes_resp,
                default_x_label="N/A", default_y_label="N/A"
            )
            return

        # ------------------------------------------------------------------
        # Observation metadata, with numeric key_index for alignment / ticks
        # ------------------------------------------------------------------
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
            for ax in (axes_misfit, axes_resp):
                ax.text(
                    0.5,
                    0.5,
                    "No matching observation indices.",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
            PlotTools.finalizePlot(
                plot_context,
                figure,
                axes_resp,
                default_x_label="Observation index",
                default_y_label="Absolute misfit",
            )
            return

        # Maps for observations (used only in responses plot)
        obs_values = {
            float(row["key_index"]):float(row["observations"])
            for row in obs.iter_rows(named=True)
        }
        obs_errors = {
            float(row["key_index"]):float(row["errors"])
            for row in obs.iter_rows(named=True)
        }

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

            # NOTE: misfit is now SIGNED, we do NOT take abs()
            misfits_long_eo = (
                misfits_wide
                .with_row_count("realization")
                .melt(
                    id_vars=["realization"],
                    variable_name="key_index",
                    value_name="misfit",
                )
                .with_columns(
                    pl.col("key_index").cast(pl.Float64),
                    pl.col("misfit"),  # keep signed value
                    pl.lit(eo.name).alias("ensemble_name"),
                    pl.lit(base_name).alias("ensemble_base"),
                )
                .filter(pl.col("key_index").is_in(keys))
            )

            misfits_long_list.append(misfits_long_eo)
            base_to_misfit_name[base_name] = eo.name

        misfits_long = (
            pl.concat(misfits_long_list, how="vertical")
            if misfits_long_list
            else None
        )

        # ------------------------------------------------------------------
        # Collect response ensembles (everything NOT ".misfits")
        # ------------------------------------------------------------------
        responses_long_list: list[pl.DataFrame] = []
        bases_with_responses: set[str] = set()

        for eo, df in response_data_columnar.items():
            if eo.name.endswith(".misfits"):
                continue

            resp_wide = df
            if "Unnamed: 0" in resp_wide.columns:
                resp_wide = resp_wide.drop("Unnamed: 0")

            base_name = eo.name

            responses_long_eo = (
                resp_wide
                .with_row_count("realization")
                .melt(
                    id_vars=["realization"],
                    variable_name="key_index",
                    value_name="response",
                )
                .with_columns(
                    pl.col("key_index").cast(pl.Float64),
                    pl.lit(eo.name).alias("ensemble_name"),
                    pl.lit(base_name).alias("ensemble_base"),
                )
                .filter(pl.col("key_index").is_in(keys))
            )

            responses_long_list.append(responses_long_eo)
            bases_with_responses.add(base_name)

        responses_long = (
            pl.concat(responses_long_list, how="vertical")
            if responses_long_list
            else None
        )

        # ------------------------------------------------------------------
        # Shared base-name ordering & colour map (SAME for both rows)
        # ------------------------------------------------------------------
        base_names = sorted(set(base_to_misfit_name.keys()) | bases_with_responses)
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colour_map = {
            base:colours[i % len(colours)]
            for i, base in enumerate(base_names)
        }

        x_positions = np.arange(len(keys))

        # ------------------------------------------------------------------
        # TOP ROW: signed misfits (boxplots per base ensemble)
        # ------------------------------------------------------------------
        legend_handles_mis: list[Line2D] = []
        global_mis_min = None
        global_mis_max = None

        if misfits_long is not None and base_names:
            group_width = 0.9
            slot_width = group_width / len(base_names)
            box_width = slot_width * 0.75

            for x_center, key in zip(x_positions, keys):
                group_left = x_center - group_width / 2.0

                for i, base in enumerate(base_names):
                    if base not in base_to_misfit_name:
                        continue  # this base has no misfit ensemble

                    colour = colour_map.get(base, "C0")
                    ens_name = base_to_misfit_name[base]

                    if x_center == x_positions[0]:
                        legend_handles_mis.append(
                            Line2D(
                                [], [],
                                marker="s",
                                linestyle="None",
                                color=colour,
                                label=ens_name,
                            )
                        )

                    sub = misfits_long.filter(
                        (pl.col("ensemble_base") == base)
                        & (pl.col("key_index") == key)
                    )
                    mis_vals = sub["misfit"].to_numpy()
                    if mis_vals.size == 0:
                        continue

                    # Track global min/max for symmetric y-axis around 0
                    local_min = float(np.min(mis_vals))
                    local_max = float(np.max(mis_vals))
                    if global_mis_min is None:
                        global_mis_min = local_min
                        global_mis_max = local_max
                    else:
                        global_mis_min = min(global_mis_min, local_min)
                        global_mis_max = max(global_mis_max, local_max)

                    box_center = group_left + slot_width * (i + 0.5)

                    bp = axes_misfit.boxplot(
                        mis_vals,
                        positions=[box_center],
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
                    axes_misfit.plot(
                        [box_center],
                        [mean_misfit],
                        marker="o",
                        markersize=5,
                        color=colour,
                    )

            # Symmetric y-axis around 0
            if global_mis_min is not None and global_mis_max is not None:
                max_abs = max(abs(global_mis_min), abs(global_mis_max))
                axes_misfit.set_ylim(-1.05 * max_abs, 1.05 * max_abs)

            axes_misfit.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
            axes_misfit.set_ylabel("Misfit")
            axes_misfit.grid(True, axis="y", linestyle=":", alpha=0.4)

            if legend_handles_mis:
                axes_misfit.legend(
                    handles=legend_handles_mis,
                    title="Ensemble (misfits)",
                    loc="upper right",
                )
        else:
            axes_misfit.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
            axes_misfit.set_ylabel("Misfit")
            axes_misfit.grid(True, axis="y", linestyle=":", alpha=0.4)

        # ------------------------------------------------------------------
        # BOTTOM ROW: responses as dot columns + compact obs dot+errorbar
        # ------------------------------------------------------------------
        if responses_long is not None:
            group_width_resp = 0.9
            if base_names:
                slot_width_resp = group_width_resp / len(base_names)
                jitter_halfwidth = slot_width_resp * 0.35
            else:
                slot_width_resp = 0.0
                jitter_halfwidth = 0.0

            for x_center, key in zip(x_positions, keys):
                group_left = x_center - group_width_resp / 2.0

                # Observation as a single dot with vertical error bar
                if (key in obs_values) and (key in obs_errors):
                    y_obs = obs_values[key]
                    y_err = obs_errors[key]
                    axes_resp.errorbar(
                        x_center,
                        y_obs,
                        yerr=y_err,
                        fmt="o",
                        color="black",
                        markersize=4,
                        capsize=3,
                        elinewidth=1.0,
                        alpha=0.9,
                        zorder=4,
                    )

                # Per-base response dots, same left-right order as misfit boxes
                for i, base in enumerate(base_names):
                    colour = colour_map.get(base, "C0")

                    sub = responses_long.filter(
                        (pl.col("ensemble_base") == base)
                        & (pl.col("key_index") == key)
                    )
                    vals = sub["response"].to_numpy()
                    if vals.size == 0:
                        continue

                    slot_center = group_left + slot_width_resp * (i + 0.5)
                    jitter = (np.random.rand(len(vals)) - 0.5) * 2.0 * jitter_halfwidth

                    axes_resp.scatter(
                        slot_center + jitter,
                        vals,
                        s=8,
                        alpha=0.6,
                        color=colour,
                    )

            # Vertical separators between observation indices (full height, NOT at ticks)
            if len(x_positions) > 1:
                separator_positions = x_positions[:-1] + 0.5  # 0.5, 1.5, 2.5, ...
                for x_sep in separator_positions:
                    axes_resp.axvline(
                        x_sep,
                        color="lightgrey",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.8,
                        zorder=0,  # keep behind dots & error bars
                    )

            axes_resp.set_xlabel("Observation index")
            axes_resp.set_ylabel("Response")
            axes_resp.grid(True, axis="y", linestyle=":", alpha=0.4)
        else:
            axes_resp.set_xlabel("Observation index")
            axes_resp.set_ylabel("Response")
            axes_resp.grid(True, axis="y", linestyle=":", alpha=0.4)

        # Remove vertical spines so they don't appear at ticks
        for ax in (axes_misfit, axes_resp):
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Shared x-ticks / labels
        axes_resp.set_xticks(x_positions)
        axes_resp.set_xticklabels([f"{int(k)}" for k in keys])
        axes_resp.tick_params(axis="x", which="both", length=0)
        axes_resp.grid(False)
        PlotTools.finalizePlot(
            plot_context,
            figure,
            axes_resp,
            default_x_label="Observation index",
            default_y_label="Response",
        )
