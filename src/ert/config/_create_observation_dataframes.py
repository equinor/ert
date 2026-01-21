from __future__ import annotations

import datetime
from collections import defaultdict
from collections.abc import Sequence
from typing import assert_never

import numpy as np
import polars as pl

from ert.validation import rangestring_to_list

from ._observations import (
    GeneralObservation,
    Observation,
    RFTObservation,
    SummaryObservation,
)
from .gen_data_config import GenDataConfig
from .parsing import (
    ErrorInfo,
    ObservationConfigError,
)
from .rft_config import RFTConfig

DEFAULT_LOCALIZATION_RADIUS = 3000


def create_observation_dataframes(
    observations: Sequence[Observation],
    gen_data_config: GenDataConfig | None,
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
                            gen_data_config,
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
    gen_data_config: GenDataConfig | None,
    general_observation: GeneralObservation,
    obs_key: str,
) -> pl.DataFrame:
    response_key = general_observation.data
    if gen_data_config is None or response_key not in gen_data_config.keys:
        raise ObservationConfigError.with_context(
            f"Problem with GENERAL_OBSERVATION {obs_key}:"
            f" No GEN_DATA with name {response_key!r} found",
            response_key,
        )
    assert isinstance(gen_data_config, GenDataConfig)
    restart = general_observation.restart
    _, report_steps = gen_data_config.get_args_for_key(response_key)

    response_report_steps = [] if report_steps is None else report_steps
    if (restart is None and response_report_steps) or (
        restart is not None and restart not in response_report_steps
    ):
        raise ObservationConfigError.with_context(
            f"The GEN_DATA node:{response_key} is not configured to load from"
            f" report step:{restart} for the observation:{obs_key}",
            response_key,
        )

    restart = 0 if restart is None else restart

    if (
        general_observation.value is not None
        and general_observation.error is not None
        and general_observation.obs_file is not None
    ):
        raise ObservationConfigError.with_context(
            "GENERAL_OBSERVATION cannot contain both VALUE/ERROR and OBS_FILE",
            context=general_observation.obs_file,
        )

    if general_observation.obs_file is not None:
        try:
            file_values = np.loadtxt(
                general_observation.obs_file, delimiter=None
            ).ravel()
        except ValueError as err:
            raise ObservationConfigError.with_context(
                f"Failed to read OBS_FILE {general_observation.obs_file}: {err}",
                general_observation.obs_file,
            ) from err
        if len(file_values) % 2 != 0:
            raise ObservationConfigError.with_context(
                "Expected even number of values in GENERAL_OBSERVATION",
                general_observation.obs_file,
            )
        values = file_values[::2]
        stds = file_values[1::2]

    else:
        assert general_observation.value is not None
        assert general_observation.error is not None
        values = np.array([general_observation.value])
        stds = np.array([general_observation.error])

    index_list = general_observation.index_list
    index_file = general_observation.index_file
    if index_list is not None and index_file is not None:
        raise ObservationConfigError.with_context(
            f"GENERAL_OBSERVATION {obs_key} has both INDEX_FILE and INDEX_LIST.",
            obs_key,
        )
    if index_file is not None:
        indices = np.loadtxt(index_file, delimiter=None, dtype=np.int32).ravel()
    elif index_list is not None:
        indices = np.array(sorted(rangestring_to_list(index_list)), dtype=np.int32)
    else:
        indices = np.arange(len(values), dtype=np.int32)

    if len({len(stds), len(values), len(indices)}) != 1:
        raise ObservationConfigError.with_context(
            f"Values ({values}), error ({stds}) and "
            f"index list ({indices}) must be of equal length",
            (
                general_observation.obs_file
                if general_observation.obs_file is not None
                else ""
            ),
        )

    if np.any(stds <= 0):
        raise ObservationConfigError.with_context(
            "Observation uncertainty must be strictly > 0", obs_key
        )
    return pl.DataFrame(
        {
            "response_key": response_key,
            "observation_key": obs_key,
            "report_step": pl.Series(
                np.full(len(indices), restart),
                dtype=pl.UInt16,
            ),
            "index": pl.Series(indices, dtype=pl.UInt16),
            "observations": pl.Series(values, dtype=pl.Float32),
            "std": pl.Series(stds, dtype=pl.Float32),
            # Location attributes will always be None for general observations, but are
            # necessary to concatenate with other observation dataframes.
            "east": pl.Series([None] * len(values), dtype=pl.Float32),
            "north": pl.Series([None] * len(values), dtype=pl.Float32),
            "radius": pl.Series([None] * len(values), dtype=pl.Float32),
        }
    )


def _handle_rft_observation(
    rft_config: RFTConfig,
    rft_observation: RFTObservation,
) -> pl.DataFrame:
    location = (rft_observation.east, rft_observation.north, rft_observation.tvd)
    if location not in rft_config.locations:
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
            "radius": pl.Series([None], dtype=pl.Float32),
            "tvd": pl.Series([location[2]], dtype=pl.Float32),
            "observations": pl.Series([rft_observation.value], dtype=pl.Float32),
            "std": pl.Series([rft_observation.error], dtype=pl.Float32),
        }
    )
