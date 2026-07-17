from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde

from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.plotting.utils.plot_context import PlotType
from ert.gui.plotting.utils.plot_tools import ConditionalAxisFormatter, PlotTools
from ert.gui.utils import truncate_experiment_name

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec

    from ert.gui.plotting.utils import PlotConfig, PlotContext
    from ert.gui.plotting.utils.plot_types import ObservationPlotLocations


class DistributionPlot:
    def __init__(self) -> None:
        self.dimensionality = 1
        self.requires_observations = False

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
        self._rug_plot = plot_context.rug_plot
        self._histogram = plot_context.histogram
        self._gkde_plot = plot_context.gkde_plot

        if not self._histogram and not self._gkde_plot and not self._rug_plot:
            figure.text(
                0.5,
                0.5,
                (
                    "No plot options selected."
                    "\n\nFrom the Distribution options (on the right-side panel),"
                    "\nplease select at least one of the following:"
                    "\nHistogram, Gaussian KDE, Rug plot"
                    "\n\nHover over the options for more information."
                ),
                ha="center",
                va="center",
                fontsize=12,
            )
            return

        if not self._histogram and not self._gkde_plot:
            # Only rug plots, no empty main plot on top
            self._plot_rug(
                figure,
                plot_context,
                ensemble_to_data_map,
                number_of_ensembles=len(plot_context.ensembles()),
                gridspec=None,
                main_plot=None,
            )
            return

        self._plot_distribution(
            figure, plot_context, ensemble_to_data_map, plot_context.by_density
        )

    def _plot_distribution(
        self,
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        by_density: bool,
    ) -> None:
        config = plot_context.plotConfig()
        number_of_ensembles = len(plot_context.ensembles())

        main_axes = self._create_main_axes(
            figure, plot_context, ensemble_to_data_map, number_of_ensembles
        )
        # Histogram uses separate y-axis ontop of
        # the density y-axis of the Gaussian KDE
        # to display count, rather than density.
        use_twin_axes = self._gkde_plot and not by_density
        histogram_axes = main_axes.twinx() if use_twin_axes else main_axes

        plot_context.x_axis = plot_context.VALUE_AXIS
        if by_density:
            plot_context.y_axis = plot_context.DENSITY_AXIS

        for (ensemble, data), color_index in zip(
            ensemble_to_data_map.items(),
            plot_context.ensembles_color_indexes(),
            strict=False,
        ):
            config.set_current_color(color_index)
            if self._gkde_plot:
                self._plot_gkde(
                    main_axes, data[0], config, log_scale=plot_context.log_scale
                )

            if self._histogram:
                self._plot_histogram(data[0], plot_context, histogram_axes, by_density)

            self._add_ensemble_legend(config, ensemble)

        if by_density:
            main_axes.set_ylim(bottom=0)
        else:
            self._scale_count_axes(main_axes, histogram_axes)

        self._finalize_axes(main_axes, histogram_axes, plot_context, by_density)

    def _create_main_axes(
        self,
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        number_of_ensembles: int,
    ) -> Axes:
        if self._rug_plot:
            gridspec = figure.add_gridspec(
                number_of_ensembles + 1,
                1,
                height_ratios=[6, *([1] * number_of_ensembles)],
                hspace=0.05,
            )
            main_axes = figure.add_subplot(gridspec[0])
            self._plot_rug(
                figure,
                plot_context,
                ensemble_to_data_map,
                number_of_ensembles,
                gridspec,
                main_axes,
            )
        else:
            main_axes = figure.add_subplot(111)
        return main_axes

    @staticmethod
    def _evaluate_kde(
        data: pd.Series,
        log_scale: bool,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        sample_range = data.max() - data.min()
        lower_bound = data.min() - 0.5 * sample_range
        upper_bound = data.max() + 0.5 * sample_range
        indexes = np.linspace(
            lower_bound if not log_scale else max(lower_bound, 1e-10),
            upper_bound,
            1000,
        )
        gkde = gaussian_kde(data.values)
        return indexes, gkde.evaluate(indexes)

    def _scale_count_axes(self, gkde_axes: Axes, histogram_axes: Axes) -> None:
        if self._gkde_plot:
            n_ticks = 6  # Ad-hoc, could be made configurable
            gkde_axes.set_ylim(bottom=0)
            histogram_axes.set_ylim(bottom=0)

            count_max = histogram_axes.get_ylim()[1]
            n_steps = math.ceil(count_max / (n_ticks - 1))
            top_int = n_steps * (n_ticks - 1)
            histogram_axes.set_ylim(0, top_int)
            histogram_axes.set_yticks(np.arange(0, top_int + 1, n_steps))

            gkde_axes.set_yticks(np.linspace(*gkde_axes.get_ylim(), n_ticks))
            gkde_axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    def _finalize_axes(
        self,
        main_axes: Axes,
        histogram_axes: Axes,
        plot_context: PlotContext,
        by_density: bool,
    ) -> None:
        if self._gkde_plot:
            self._add_gkde_legend(plot_context.plotConfig())
        if by_density:
            set_ylabel_by_config(plot_context, main_axes, "Density")
        else:
            main_axes.set_xlabel(plot_context.plotConfig().x_label() or "Value")
            if self._gkde_plot:
                histogram_axes.set_ylabel("Count (Histogram)")
                main_axes.set_ylabel("Density (Gaussian KDE)")
            else:
                set_ylabel_by_config(plot_context, main_axes, "Count")

        plot_context.plot_type = PlotType.BAR

        PlotTools.set_title(main_axes, plot_context)

        axes_to_clean = [main_axes] if by_density else [main_axes, histogram_axes]
        for axes in axes_to_clean:
            PlotTools.remove_spines(axes, ["right", "left", "top"])

        PlotTools.show_grid(main_axes, plot_context)
        PlotTools.show_legend(main_axes, plot_context)

    def _plot_gkde(
        self, axes: Axes, data: pd.Series, config: PlotConfig, log_scale: bool
    ) -> None:
        if _array_is_empty_or_non_numeric(data) or _array_is_constant(data):
            return
        indexes, evaluated = self._evaluate_kde(data, log_scale=log_scale)
        if log_scale:
            axes.set_xscale("log")
        axes.plot(indexes, evaluated, color=config.current_color())

    def _plot_histogram(
        self,
        data: pd.Series,
        plot_context: PlotContext,
        histogram_axes: Axes,
        by_density: bool,
    ) -> None:
        if _array_is_empty_or_non_numeric(data):
            return

        config = plot_context.plotConfig()
        bins: str | Sequence[float]
        if plot_context.log_scale:
            log_edges = np.histogram_bin_edges(np.log10(data), bins="sqrt")
            bins = (10**log_edges).tolist()
            histogram_axes.set_xscale("log")
        else:
            bins = "sqrt"
            histogram_axes.set_xscale("linear")
            histogram_axes.xaxis.set_major_formatter(ConditionalAxisFormatter())

        histogram_axes.hist(
            data,
            bins=bins,
            density=by_density,
            alpha=0.3,
            color=config.current_color(),
        )

    def _plot_rug(
        self,
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        number_of_ensembles: int,
        gridspec: GridSpec | None = None,
        main_plot: Axes | None = None,
    ) -> None:
        config = plot_context.plotConfig()
        only_rug = main_plot is None

        if only_rug:
            gridspec = figure.add_gridspec(number_of_ensembles, 1, hspace=0.02)

        # In only_rug mode rugs share x with each other; otherwise with main_plot
        share_ref = main_plot
        rug_plots: list[Axes] = []
        for i in range(number_of_ensembles):
            row_index = i if only_rug else i + 1
            axes = figure.add_subplot(gridspec[row_index], sharex=share_ref)  # type: ignore
            share_ref = share_ref or axes
            rug_plots.append(axes)

        for index, ((ensemble, data), color_index) in enumerate(
            zip(
                ensemble_to_data_map.items(),
                plot_context.ensembles_color_indexes(),
                strict=False,
            )
        ):
            if _array_is_empty_or_non_numeric(data[0]):
                continue

            rug = rug_plots[index]
            config.set_current_color(color_index)
            rug.plot(
                data[0],
                np.zeros(len(data[0])),
                marker="|",
                markersize=15,
                linestyle="",
                color=config.current_color(),
            )
            rug.axhline(
                y=0,
                color="grey",
                linewidth=0.8,
            )
            label = (
                f"{truncate_experiment_name(ensemble.experiment_name)}"
                f" : {ensemble.name}"
            )

            if only_rug:
                rug.legend(
                    [
                        Line2D(
                            [],
                            [],
                            marker="s",
                            linestyle="None",
                            color=config.current_color(),
                            label=label,
                        )
                    ],
                    [label],
                    numpoints=1,
                )
            rug.yaxis.set_visible(False)
            rug.xaxis.set_visible(
                only_rug or index == number_of_ensembles - 1
            )  # x-axis on all if no mainplot, otherwise only on last rug plot
            if only_rug or index == number_of_ensembles - 1:
                rug.tick_params(axis="x", labelbottom=True)
            rug.set_ylim(-0.15, 0.5)
            PlotTools.remove_spines(rug, ["top", "right", "left", "bottom"])
            if plot_context.log_scale:
                rug.set_xscale("log")

        if only_rug:
            PlotTools.set_title(rug_plots[0], plot_context)
        else:
            rug_plots[0].set_title("Realization distribution")

    def _add_ensemble_legend(
        self, config: PlotConfig, ensemble: EnsembleObject
    ) -> None:
        label = (
            f"{truncate_experiment_name(ensemble.experiment_name)} : {ensemble.name}"
        )
        config.add_legend_item(
            label,
            Line2D(
                [],
                [],
                marker="s",
                linestyle="None",
                color=config.current_color(),
                label=label,
            ),
        )

    def _add_gkde_legend(self, config: PlotConfig) -> None:
        config.add_legend_item(
            "Gaussian KDE",
            Line2D(
                [],
                [],
                linestyle="-",
                color="grey",
                label="Gaussian KDE",
            ),
        )


def set_ylabel_by_config(plot_context: PlotContext, axes: Axes, y_label: str) -> None:
    config = plot_context.plotConfig()
    if config.x_label() is None:
        config.set_x_label("Value")
    if config.y_label() is None:
        config.set_y_label(y_label)
    PlotTools.set_labels_for_axes_from_context(axes, plot_context)


def _array_is_constant(data: pd.Series | pd.DataFrame) -> bool:
    array = data.to_numpy()
    return array.shape[0] == 0 or (array[0] == array).all()


def _array_is_empty_or_non_numeric(data: pd.Series | pd.DataFrame) -> bool:
    return data.empty or not pd.api.types.is_numeric_dtype(data)
