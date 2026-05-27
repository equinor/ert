from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ert.field_utils.field_utils import gaspari_cohn
from ert.gui.tools.plot.plot_types import LocalizationProvider, ObservationPlotLocations

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent

    from ert.gui.tools.plot.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.tools.plot.plottery import PlotContext


class StdDevPlot:
    _PICK_RADIUS_PIXELS = 8.0
    # Show localization rho as increasing opacity over a fixed orange color.
    _LOCALIZATION_RHO_CMAP = LinearSegmentedColormap.from_list(
        "ert_localization_rho",
        [(*to_rgba("#ffb000")[:3], 0.0), (*to_rgba("#ffb000")[:3], 0.35)],
    )

    def __init__(self) -> None:
        self.dimensionality = 3
        self.requires_observations = False
        self._selected_observation_artists: dict[Any, list[Any]] = {}
        self._observation_click_callback_ids: dict[Any, list[int]] = {}

    def plot(
        self,
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_data: dict[str, npt.NDArray[np.float32]],
        obs_loc: ObservationPlotLocations | None,
        localization_provider: LocalizationProvider | None = None,
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None:
        ensembles = plot_context.ensembles()
        self._clear_observation_callbacks(figure)
        self._selected_observation_artists.clear()
        if (ensemble_count := len(ensembles)) == 0:
            return
        layer = plot_context.layer
        if layer is not None:
            vmin: float = np.inf
            vmax: float = -np.inf
            heatmaps = []
            boxplot_axes = []

            # Adjust height_ratios to reduce space between plots
            figure.set_layout_engine("constrained")
            gridspec = figure.add_gridspec(2, ensemble_count, hspace=0.2)

            for i, ensemble in enumerate(reversed(ensembles), start=1):
                ax_heat = figure.add_subplot(gridspec[0, i - 1])
                ax_box = figure.add_subplot(gridspec[1, i - 1])
                data = std_dev_data[ensemble.id]
                if data.size == 0:
                    ax_heat.set_axis_off()
                    ax_box.set_axis_off()
                    ax_heat.text(
                        0.5,
                        0.5,
                        f"No data for {ensemble.experiment_name} : {ensemble.name}",
                        ha="center",
                        va="center",
                    )
                else:
                    vmin = min(vmin, float(np.min(data)))
                    vmax = max(vmax, float(np.max(data)))

                    im = ax_heat.imshow(data.T, cmap="viridis", aspect="equal")

                    if obs_loc is not None:
                        xs = obs_loc.x.astype(np.float32) - 0.5
                        ys = obs_loc.y.astype(np.float32) - 0.5

                        ax_heat.scatter(
                            xs,
                            ys,
                            c="tab:orange",
                            marker="o",
                            edgecolors="black",
                            linewidths=0.5,
                            label="Observations",
                            zorder=4,
                        )
                        callback_id = figure.canvas.mpl_connect(
                            "button_press_event",
                            partial(
                                self._on_observation_click,
                                ax=ax_heat,
                                data_shape=data.shape,
                                parameter_key=plot_context.key(),
                                ensemble_id=ensemble.id,
                                obs_loc=obs_loc,
                                localization_provider=localization_provider,
                            ),
                        )
                        self._observation_click_callback_ids.setdefault(
                            figure.canvas, []
                        ).append(callback_id)
                    heatmaps.append(im)

                    ax_box.boxplot(data.flatten(), orientation="vertical", widths=0.5)
                    boxplot_axes.append(ax_box)

                    min_value = np.min(data)
                    mean_value = np.mean(data)
                    max_value = np.max(data)

                    annotation_text = (
                        f"Min: {min_value:.2f}\n"
                        f"Mean: {mean_value:.2f}\nMax: {max_value:.2f}"
                    )
                    ax_box.annotate(
                        annotation_text,
                        xy=(1, 1),  # Changed from (0, 1) to (1, 1)
                        xycoords="axes fraction",
                        ha="right",  # Changed from 'left' to 'right'
                        va="top",
                        fontsize=8,
                        fontweight="bold",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "black",
                            "boxstyle": "round,pad=0.2",
                        },
                    )

                    ax_box.spines["top"].set_visible(False)
                    ax_box.spines["right"].set_visible(False)
                    ax_box.spines["bottom"].set_visible(False)
                    ax_box.spines["left"].set_visible(True)

                    ax_box.set_xticks([])
                    ax_box.set_xticklabels([])

                    ax_heat.set_ylabel("")
                    ax_box.set_ylabel(
                        "Standard deviation", fontsize=8
                    )  # Reduced font size

                    self._colorbar(im)

                ax_heat.set_title(
                    f"{ensemble.experiment_name} : {ensemble.name} layer={layer}",
                    wrap=True,
                    fontsize=10,  # Reduced font size
                )

            norm = plt.Normalize(vmin, vmax)
            for im in heatmaps:
                im.set_norm(norm)

            padding = 0.05 * (vmax - vmin)
            if padding > 0.0:
                for ax_box in boxplot_axes:
                    ax_box.set_ylim(vmin - padding, vmax + padding)

    @staticmethod
    def _localization_overlay(
        data_shape: tuple[int, ...],
        observation: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        x, y, radius_x, radius_y = observation[:4].astype(np.float32)
        nx, ny = data_shape[:2]
        x_grid, y_grid = np.meshgrid(
            np.linspace(0, nx - 1, nx, dtype=np.float32),
            np.linspace(0, ny - 1, ny, dtype=np.float32),
        )
        distance = np.hypot(
            (x_grid - (x - 0.5)) / radius_x,
            (y_grid - (y - 0.5)) / radius_y,
        )
        taper = gaspari_cohn(distance).astype(np.float32)
        overlay = np.zeros((*taper.shape, 4), dtype=np.float32)
        overlay[..., :3] = to_rgba("#ffb000")[:3]
        overlay[..., 3] = taper * 0.35
        return overlay

    def _on_observation_click(
        self,
        event: MouseEvent,
        ax: Axes,
        data_shape: tuple[int, ...],
        parameter_key: str,
        ensemble_id: str,
        obs_loc: ObservationPlotLocations | None,
        localization_provider: LocalizationProvider | None,
    ) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            self._clear_selected_observation(ax)
            if event.canvas is not None:
                event.canvas.draw_idle()
            return

        assert obs_loc is not None
        obs_xy = np.column_stack((obs_loc.x, obs_loc.y)).astype(np.float32) - 0.5
        display_points = ax.transData.transform(obs_xy)
        distances = np.hypot(
            display_points[:, 0] - event.x, display_points[:, 1] - event.y
        )
        index = int(np.argmin(distances))
        if distances[index] > self._PICK_RADIUS_PIXELS:
            self._clear_selected_observation(ax)
            if event.canvas is not None:
                event.canvas.draw_idle()
            return

        self._draw_selected_observation(
            ax,
            data_shape,
            parameter_key,
            ensemble_id,
            index,
            obs_loc,
            localization_provider,
        )
        if event.canvas is not None:
            event.canvas.draw_idle()

    def _clear_selected_observation(self, ax: Axes) -> None:
        for artist in self._selected_observation_artists.pop(ax, []):
            artist.remove()

    def _clear_observation_callbacks(self, figure: Figure) -> None:
        for callback_id in self._observation_click_callback_ids.pop(figure.canvas, []):
            figure.canvas.mpl_disconnect(callback_id)

    def _draw_selected_observation(
        self,
        ax: Axes,
        data_shape: tuple[int, ...],
        parameter_key: str,
        ensemble_id: str,
        observation_index: int,
        obs_loc: ObservationPlotLocations | None,
        localization_provider: LocalizationProvider | None = None,
    ) -> None:
        self._clear_selected_observation(ax)
        assert obs_loc is not None
        radius_x = obs_loc.radius_x[observation_index]
        radius_y = obs_loc.radius_y[observation_index]
        if radius_x <= 0.0 or radius_y <= 0.0:
            return
        rho = self._rho_for_observation(
            parameter_key,
            ensemble_id,
            observation_index,
            obs_loc,
            localization_provider,
        )
        overlay = ax.imshow(
            rho.T
            if rho is not None
            else self._localization_overlay(
                data_shape,
                np.array(
                    [
                        obs_loc.x[observation_index],
                        obs_loc.y[observation_index],
                        radius_x,
                        radius_y,
                    ],
                    dtype=np.float32,
                ),
            ),
            aspect="equal",
            cmap=self._LOCALIZATION_RHO_CMAP if rho is not None else None,
            interpolation="bilinear",
            extent=(-0.5, data_shape[0] - 0.5, data_shape[1] - 0.5, -0.5),
            vmin=0.0 if rho is not None else None,
            vmax=1.0 if rho is not None else None,
            zorder=2,
        )
        ax.set_xlim(-0.5, data_shape[0] - 0.5)
        ax.set_ylim(data_shape[1] - 0.5, -0.5)
        self._selected_observation_artists[ax] = [overlay]

    @staticmethod
    def _rho_for_observation(
        parameter_key: str,
        ensemble_id: str,
        observation_index: int,
        obs_loc: ObservationPlotLocations | None,
        localization_provider: LocalizationProvider | None,
    ) -> npt.NDArray[np.float32] | None:
        if localization_provider is None or obs_loc is None:
            return None
        return localization_provider(
            parameter_key,
            ensemble_id,
            obs_loc.observation_key[observation_index],
            obs_loc.observation_index[observation_index],
        )

    @staticmethod
    def _colorbar(mappable: Any) -> Any:
        last_axes = plt.gca()
        ax = mappable.axes
        assert ax is not None
        fig = ax.figure
        assert fig is not None
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        plt.sca(last_axes)
        return cbar
