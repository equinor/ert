from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.plotting.utils.plot_tools import ConditionalAxisFormatter, PlotTools
from ert.gui.utils import truncate_experiment_name

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ert.gui.plotting.utils import PlotConfig, PlotContext
    from ert.gui.plotting.utils.plot_types import ObservationPlotLocations


class CrossPlot:
    def __init__(self) -> None:
        self.dimensionality = 2
        self.requires_observations = True

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
        plotCross(figure, plot_context, ensemble_to_data_map, observation_data)


def plotCross(
    figure: Figure,
    plot_context: PlotContext,
    ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
    observation_data: pd.DataFrame,
) -> None:
    config = plot_context.plotConfig()
    axes = figure.add_subplot(111)

    plot_context.x_axis = plot_context.VALUE_AXIS
    plot_context.y_axis = plot_context.VALUE_AXIS
    plot_context.deactivate_date_support()

    if observation_data.empty:
        axes.text(0.5, 0.5, "No observations available", ha="center", va="center")
        axes.set_axis_off()
        return

    obs_by_key_index = _observations_by_key_index(observation_data)

    all_obs: list[np.ndarray] = []
    all_resp: list[np.ndarray] = []

    for (ensemble, data), color_index in zip(
        ensemble_to_data_map.items(),
        plot_context.ensembles_color_indexes(),
        strict=False,
    ):
        config.set_current_color(color_index)

        if data.empty:
            continue

        obs_values, resp_values = _match_obs_to_responses(data, obs_by_key_index)
        if obs_values.size == 0:
            continue

        label = (
            f"{truncate_experiment_name(ensemble.experiment_name)} : {ensemble.name}"
        )
        _plot_cross(axes, config, obs_values, resp_values, label)

        all_obs.append(obs_values)
        all_resp.append(resp_values)

    if all_obs:
        _plot_identity_line(
            axes, config, np.concatenate(all_obs), np.concatenate(all_resp)
        )

    axes.xaxis.set_major_formatter(ConditionalAxisFormatter())
    axes.yaxis.set_major_formatter(ConditionalAxisFormatter())

    if plot_context.log_scale:
        axes.set_xscale("log")
        axes.set_yscale("log")

    PlotTools.finalize_plot(
        plot_context,
        figure,
        axes,
        default_x_label="Observation",
        default_y_label="Response",
    )


def _observations_by_key_index(observation_data: pd.DataFrame) -> pd.Series:
    obs_values = pd.to_numeric(
        observation_data.loc["OBS"].to_numpy(), errors="coerce"
    ).astype(np.float32)
    raw_key_index = observation_data.loc["key_index"].to_numpy()
    key_index = _to_matchable_key(raw_key_index)
    series = pd.Series(obs_values, index=key_index)
    return series[~series.index.duplicated(keep="first")].dropna()


def _match_obs_to_responses(
    ensemble_data: pd.DataFrame, obs_by_key_index: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    responses = ensemble_data.copy()
    responses.columns = _to_matchable_key(responses.columns.to_numpy())

    common_keys = obs_by_key_index.index.intersection(responses.columns)
    if len(common_keys) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    obs_matched = obs_by_key_index.loc[common_keys].to_numpy()
    resp_matched = (
        responses[common_keys]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy()
        .astype(np.float32)
    )

    n_realizations = resp_matched.shape[0]
    obs_flat = np.tile(obs_matched, n_realizations)
    resp_flat = resp_matched.reshape(-1)

    mask = ~np.isnan(obs_flat) & ~np.isnan(resp_flat)
    return obs_flat[mask], resp_flat[mask]


def _to_matchable_key(values: np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.notna().all():
        return numeric.astype(np.float32).to_numpy()
    return pd.Index(values).astype(str).to_numpy()


def _plot_cross(
    axes: Axes,
    plot_config: PlotConfig,
    obs_values: np.ndarray,
    resp_values: np.ndarray,
    label: str,
) -> None:
    style = plot_config.distribution_style()

    lines = axes.plot(
        obs_values,
        resp_values,
        color=style.color,
        alpha=style.alpha,
        marker=style.marker or "o",
        markersize=style.size,
        linestyle="",
        label=label,
    )

    if lines:
        plot_config.add_legend_item(label, lines[0])


def _plot_identity_line(
    axes: Axes,
    plot_config: PlotConfig,
    obs_values: np.ndarray,
    resp_values: np.ndarray,
) -> None:
    lo = float(min(obs_values.min(), resp_values.min()))
    hi = float(max(obs_values.max(), resp_values.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return

    lines = axes.plot(
        [lo, hi],
        [lo, hi],
        color="red",
        linestyle="-",
        linewidth=1,
        zorder=0,
    )
    if lines:
        plot_config.add_legend_item("y = x", lines[0])
