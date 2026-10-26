from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import polars as pl

from ert.config._reservoir_data_utils import SeismicData
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
        assert key_def is not None, "key_def must be provided"
        response_type = key_def.metadata["data_origin"]
        plotCross(
            figure,
            plot_context,
            ensemble_to_data_map,
            observation_data,
            key_def=key_def,
            response_type=response_type,
        )


def plotCross(
    figure: Figure,
    plot_context: PlotContext,
    ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
    observation_data: pd.DataFrame,
    key_def: PlotApiKeyDefinition,
    response_type: str,
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

        obs_values, resp_values = _match_obs_to_responses(
            data,
            obs_data=observation_data,
            key_def=key_def,
            response_type=response_type,
        )
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


def _parse_coordinate_pairs(values: Iterable[Any]) -> list[tuple[float, float]]:
    pairs = []
    for value in values:
        east, north = str(value).strip("()").split(",")
        pairs.append((float(east), float(north)))
    return pairs


def _join_obs_with_response_data(
    ensemble_df: pd.DataFrame,
    observation_data: pd.DataFrame,
    response_type: Literal["summary", "gen_data", "rft", "breakthrough", "seismic"],
    key_def: PlotApiKeyDefinition,
) -> pl.DataFrame:
    if response_type in {"summary", "breakthrough"}:
        key_index_with_correct_dtype = pl.col("key_index").str.to_datetime(strict=False)
    elif response_type in {"gen_data", "rft"}:
        key_index_with_correct_dtype = (
            pl.col("key_index").cast(pl.Float32).cast(pl.UInt16)
        )
    elif response_type == "seismic":
        key_index_with_correct_dtype = pl.col("key_index")
    else:
        raise ValueError(f"Unsupported response_type: {response_type}")

    obs_df = (
        pl.from_pandas(observation_data.T)
        .rename({"OBS": "observation", "STD": "error"})
        .with_columns(key_index_with_correct_dtype)
    )

    east, north = zip(*_parse_coordinate_pairs(obs_df["key_index"]), strict=True)
    obs_df = obs_df.with_columns(pl.Series("east", east), pl.Series("north", north))
    obs_df = obs_df.with_columns(pl.lit(key_def.key).alias("response_key"))

    response_df = (
        pl.from_pandas(ensemble_df, include_index=True)
        .unpivot(
            index=ensemble_df.index.name,
            variable_name="key_index",
            value_name="response",
        )
        .with_columns(key_index_with_correct_dtype)
    )
    east, north = zip(*_parse_coordinate_pairs(response_df["key_index"]), strict=True)
    response_df = response_df.with_columns(
        pl.Series("east", east), pl.Series("north", north)
    )
    response_df = response_df.with_columns(pl.lit(key_def.key).alias("response_key"))
    if response_type == "seismic":
        response_df = SeismicData.use_observation_locations_in_respective_responses(
            response_df, obs_df
        )
        joint_df = response_df.join(obs_df, on=["east", "north"], how="inner")
    else:
        joint_df = response_df.join(obs_df, on="key_index", how="inner")
    return joint_df


def _match_obs_to_responses(
    ensemble_data: pd.DataFrame,
    obs_data: pd.DataFrame,
    key_def: PlotApiKeyDefinition,
    response_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    df = _join_obs_with_response_data(
        ensemble_data,
        observation_data=obs_data,
        key_def=key_def,
        response_type=response_type,
    )
    if df.is_empty():
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return df["observation"].to_numpy(), df["response"].to_numpy()


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
