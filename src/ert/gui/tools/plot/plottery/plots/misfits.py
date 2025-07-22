from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import seaborn as sns  # Used for statistical plots like boxplots and histograms

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
        ensemble_to_data_map: dict[EnsembleObject, pl.DataFrame],
        observation_data: pl.DataFrame,
        std_dev_images: dict[str, npt.NDArray[np.float32]],
    ) -> None:
        config = plot_context.plotConfig()
        axes = figure.add_subplot(111)
        response_key = plot_context.key()
        misfit_plot_mode = config.getMisfitsPlotMode()

        if observation_data.is_empty():
            axes.text(
                0.5,
                0.5,
                "No observations available for this key.",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axes.transAxes,
                fontsize=12,
            )
            PlotTools.finalizePlot(
                plot_context, figure, axes, default_x_label="N/A", default_y_label="N/A"
            )
            return

        if (
            observation_data.columns[0] == "STD"
            and observation_data.columns[1] == "OBS"
        ):
            observation_data = observation_data.transpose(include_header=True)
            observation_data = observation_data.rename(
                {"OBS": "observations", "STD": "std"}
            )
            observation_data = observation_data.with_columns(
                pl.col("observations").cast(pl.Float64),
                pl.col("std").cast(pl.Float64),
                pl.col("""column""").alias("x_axis").str.to_datetime(),
            )
        elif "x_axis" not in observation_data.columns:
            observation_data = observation_data.with_columns(
                pl.Series(
                    "x_axis", observation_data.select_at_idx(0).to_series()
                ).str.to_datetime()
            )

        all_misfits_data = []

        for ensemble, response_df in ensemble_to_data_map.items():
            if not ensemble.name.endswith(".misfits") or response_df.is_empty():
                continue

            if "Unnamed: 0" in response_df.columns:
                response_df = response_df.drop("Unnamed: 0")

            response_df = response_df.transpose(include_header=True)
            response_df = response_df.with_columns(
                pl.col("column").str.to_datetime().alias("x_axis")
            )

            joined = response_df.join(observation_data, on="x_axis", how="inner")
            if joined.is_empty():
                continue

            misfits_cols = []
            for col in response_df.columns:
                if col in {"x_axis", "observations", "std", "column"}:
                    continue
                misfit = (
                    ((joined[col] - joined["observations"]) / joined["std"]).pow(2)
                ).alias(f"Realization {col}")
                misfits_cols.append(misfit)

            if not misfits_cols:
                continue

            misfit_df = pl.DataFrame(misfits_cols + [joined["x_axis"]])
            misfit_df = misfit_df.with_columns(pl.lit(ensemble.name).alias("Ensemble"))
            all_misfits_data.append(misfit_df)

        if not all_misfits_data:
            axes.text(
                0.5,
                0.5,
                "No valid misfits could be calculated.",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axes.transAxes,
                fontsize=12,
            )
            PlotTools.finalizePlot(
                plot_context, figure, axes, default_x_label="N/A", default_y_label="N/A"
            )
            return

        combined_df = pl.concat(all_misfits_data, how="vertical")
        combined_df = combined_df.rename({"x_axis": "Time"})
        time_column_name = "Time"

        combined_df_pd = combined_df.to_pandas()

        if plot_context.isDateSupportActive():
            plot_context.x_axis = plot_context.DATE_AXIS
        else:
            plot_context.x_axis = plot_context.INDEX_AXIS

        misfit_value_columns = [
            col for col in combined_df_pd.columns if col.startswith("Realization")
        ]

        if misfit_plot_mode == "Univariate":
            misfits_long_format = combined_df_pd.melt(
                id_vars=["Ensemble", time_column_name],
                value_vars=misfit_value_columns,
                var_name="Realization",
                value_name="Misfit Value",
            ).sort_values(by=time_column_name)

            sns.boxplot(
                x=time_column_name,
                y="Misfit Value",
                hue="Ensemble",
                data=misfits_long_format,
                ax=axes,
                palette="tab10",
                showfliers=False,
            )

            plot_context.y_axis = plot_context.VALUE_AXIS
            default_x_label = time_column_name
            default_y_label = "Misfit Value"

            if len(combined_df_pd[time_column_name].unique()) > 10:
                figure.autofmt_xdate(rotation=45)

            axes.set_title(f"Misfit Boxplots for {response_key}")
            if len(misfits_long_format["Ensemble"].unique()) > 1:
                axes.legend(title="Ensemble")

        elif misfit_plot_mode == "Summary":
            all_misfit_values = combined_df_pd[misfit_value_columns].values.flatten()
            all_misfit_values = all_misfit_values[~np.isnan(all_misfit_values)]

            if all_misfit_values.size == 0:
                axes.text(
                    0.5,
                    0.5,
                    "No valid misfits to display in histogram.",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axes.transAxes,
                    fontsize=12,
                )
                PlotTools.finalizePlot(
                    plot_context,
                    figure,
                    axes,
                    default_x_label="N/A",
                    default_y_label="N/A",
                )
                return

            sns.histplot(all_misfit_values, kde=True, ax=axes)

            plot_context.x_axis = plot_context.VALUE_AXIS
            plot_context.y_axis = plot_context.COUNT_AXIS
            default_x_label = "Misfit Value"
            default_y_label = "Frequency"
            axes.set_title(f"Misfit Histogram for {response_key}")

        else:
            axes.text(
                0.5,
                0.5,
                f"Invalid misfit plot mode: {misfit_plot_mode}.",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axes.transAxes,
                fontsize=12,
            )
            PlotTools.finalizePlot(
                plot_context, figure, axes, default_x_label="N/A", default_y_label="N/A"
            )
            return

        PlotTools.finalizePlot(
            plot_context,
            figure,
            axes,
            default_x_label=default_x_label,
            default_y_label=default_y_label,
        )
