from __future__ import annotations

import datetime
from collections import defaultdict
from collections.abc import Sequence
from typing import assert_never

import polars as pl

from ._observations import (
    BreakthroughObservation,
    GeneralObservation,
    Observation,
    RFTObservation,
    SummaryObservation,
)
from .parsing import (
    ErrorInfo,
    ObservationConfigError,
)
from .rft_config import RFTConfig

DEFAULT_LOCALIZATION_RADIUS = 3000


def create_observation_dataframes(
    observations: Sequence[Observation],
    rft_config: RFTConfig | None,
) -> dict[str, pl.DataFrame]:
    if not observations:
        return {}

    config_errors: list[ErrorInfo] = []
    grouped: dict[str, list[pl.DataFrame]] = defaultdict(list)
    for obs in observations:
        try:
            match obs:
                case SummaryObservation():
                    grouped["summary"].append(
                        _handle_summary_observation(
                            obs,
                            obs.name,
                        )
                    )
                case GeneralObservation():
                    grouped["gen_data"].append(
                        _handle_general_observation(
                            obs,
                            obs.name,
                        )
                    )
                case RFTObservation():
                    if rft_config is None:
                        raise TypeError(
                            "create_observation_dataframes requires "
                            "rft_config is not None when using RFTObservation"
                        )
                    grouped["rft"].append(_handle_rft_observation(rft_config, obs))
                case BreakthroughObservation():
                    grouped["breakthrough"].append(
                        _handle_breakthrough_observation(
                            obs,
                        )
                    )
                case default:
                    assert_never(default)
        except ObservationConfigError as err:
            config_errors.extend(err.errors)

    if config_errors:
        raise ObservationConfigError.from_collected(config_errors)

    datasets: dict[str, pl.DataFrame] = {}

    for name, dfs in grouped.items():
        non_empty_dfs = [df for df in dfs if not df.is_empty()]
        if len(non_empty_dfs) > 0:
            ds = pl.concat(non_empty_dfs).sort("observation_key")
            if "time" in ds:
                ds = ds.sort(by="time")

            datasets[name] = ds
    return datasets


def _has_localization(summary_dict: SummaryObservation) -> bool:
    return summary_dict.east is not None and summary_dict.north is not None


def _handle_summary_observation(
    summary_dict: SummaryObservation,
    obs_key: str,
) -> pl.DataFrame:
    summary_key = summary_dict.key
    value = summary_dict.value
    std_dev = summary_dict.error
    date = datetime.datetime.fromisoformat(summary_dict.date)

    localization_radius = (
        summary_dict.radius or DEFAULT_LOCALIZATION_RADIUS
        if _has_localization(summary_dict)
        else None
    )

    return pl.DataFrame(
        {
            "response_key": [summary_key],
            "observation_key": [obs_key],
            "time": pl.Series([date]).dt.cast_time_unit("ms"),
            "observations": pl.Series([value], dtype=pl.Float32),
            "std": pl.Series([std_dev], dtype=pl.Float32),
            "east": pl.Series([summary_dict.east], dtype=pl.Float32),
            "north": pl.Series([summary_dict.north], dtype=pl.Float32),
            "radius": pl.Series([localization_radius], dtype=pl.Float32),
        }
    )


def _handle_general_observation(
    general_observation: GeneralObservation,
    obs_key: str,
) -> pl.DataFrame:
    response_key = general_observation.data
    restart = general_observation.restart

    if general_observation.error <= 0:
        raise ObservationConfigError.with_context(
            "Observation uncertainty must be strictly > 0", obs_key
        )

    return pl.DataFrame(
        {
            "response_key": [response_key],
            "observation_key": [general_observation.name],
            "report_step": pl.Series([restart], dtype=pl.UInt16),
            "index": pl.Series([general_observation.index], dtype=pl.UInt16),
            "observations": pl.Series([general_observation.value], dtype=pl.Float32),
            "std": pl.Series([general_observation.error], dtype=pl.Float32),
            "east": pl.Series([general_observation.east], dtype=pl.Float32),
            "north": pl.Series([general_observation.north], dtype=pl.Float32),
            "radius": pl.Series([general_observation.radius], dtype=pl.Float32),
        }
    )


def _handle_rft_observation(
    rft_config: RFTConfig,
    rft_observation: RFTObservation,
) -> pl.DataFrame:
    location = (rft_observation.east, rft_observation.north, rft_observation.tvd)
    zones = {zone for zones in rft_config.zonemap.values() for zone in zones}
    if location not in rft_config.locations:
        if (zone := rft_observation.zone) is not None:
            if zone not in zones:
                raise ObservationConfigError(
                    f"The RFT_OBSERVATION {rft_observation.name} was given "
                    f"zone {zone} but no such zone exists in the ZONEMAP."
                )
            rft_config.locations.append((location, zone))
        else:
            rft_config.locations.append(location)

    data_to_read = rft_config.data_to_read
    if rft_observation.well not in data_to_read:
        rft_config.data_to_read[rft_observation.well] = {}

    well_dict = data_to_read[rft_observation.well]
    if rft_observation.date not in well_dict:
        well_dict[rft_observation.date] = []

    property_list = well_dict[rft_observation.date]
    if rft_observation.property not in property_list:
        property_list.append(rft_observation.property)

    if rft_observation.error <= 0.0:
        raise ObservationConfigError.with_context(
            "Observation uncertainty must be strictly > 0", rft_observation.well
        )

    return pl.DataFrame(
        {
            "response_key": (
                f"{rft_observation.well}:"
                f"{rft_observation.date}:"
                f"{rft_observation.property}"
            ),
            "observation_key": rft_observation.name,
            "east": pl.Series([location[0]], dtype=pl.Float32),
            "north": pl.Series([location[1]], dtype=pl.Float32),
            "tvd": pl.Series([location[2]], dtype=pl.Float32),
            "zone": pl.Series([rft_observation.zone], dtype=pl.String),
            "observations": pl.Series([rft_observation.value], dtype=pl.Float32),
            "std": pl.Series([rft_observation.error], dtype=pl.Float32),
            "radius": pl.Series([None], dtype=pl.Float32),
        }
    )


def _handle_breakthrough_observation(
    obs_config: BreakthroughObservation,
) -> pl.DataFrame:
    error_hours = obs_config.error * 24
    return pl.DataFrame(
        {
            "observation_key": obs_config.name,
            "response_key": f"BREAKTHROUGH:{obs_config.key}",
            "time": pl.Series([obs_config.date]).dt.cast_time_unit("ms"),
            "observations": pl.Series([0], dtype=pl.Float32),
            "threshold": obs_config.threshold,
            "std": pl.Series([error_hours], dtype=pl.Float32),
            "east": pl.Series([obs_config.east], dtype=pl.Float32),
            "north": pl.Series([obs_config.north], dtype=pl.Float32),
            "radius": pl.Series([obs_config.radius], dtype=pl.Float32),
        }
    )
