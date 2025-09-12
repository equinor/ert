from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ert.summary_key_type import history_key
from ert.validation import rangestring_to_list

from ._observation_declaration import (
    ConfContent,
    DateValues,
    ErrorValues,
    GenObsValues,
    HistoryValues,
    SummaryValues,
)
from .ensemble_config import EnsembleConfig
from .gen_data_config import GenDataConfig
from .parsing import (
    ConfigWarning,
    ErrorInfo,
    HistorySource,
    ObservationConfigError,
)
from .summary_observation import SummaryObservation

if TYPE_CHECKING:
    import numpy.typing as npt


DEFAULT_TIME_DELTA = timedelta(seconds=30)


@dataclass(eq=False)
class _GenObservation:
    values: npt.NDArray[np.float64]
    stds: npt.NDArray[np.float64]
    indices: npt.NDArray[np.int32]
    std_scaling: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        for val in self.stds:
            if val <= 0:
                raise ValueError("Observation uncertainty must be strictly > 0")


@dataclass
class _GenObs:
    observation_key: str
    data_key: str
    observations: dict[int, _GenObservation]

    def to_dataset(self) -> pl.DataFrame:
        dataframes = []
        for time_step, node in self.observations.items():
            dataframes.append(
                pl.DataFrame(
                    {
                        "response_key": self.data_key,
                        "observation_key": self.observation_key,
                        "report_step": pl.Series(
                            np.full(len(node.indices), time_step),
                            dtype=pl.UInt16,
                        ),
                        "index": pl.Series(node.indices, dtype=pl.UInt16),
                        "observations": pl.Series(node.values, dtype=pl.Float32),
                        "std": pl.Series(node.stds, dtype=pl.Float32),
                    }
                )
            )

        combined = pl.concat(dataframes)
        return combined


@dataclass
class _SummaryObs:
    observation_key: str
    data_key: str
    observations: dict[datetime, SummaryObservation]

    def to_dataset(self) -> pl.DataFrame:
        observations = []
        actual_response_key = self.observation_key
        actual_observation_keys = []
        errors = []
        dates = list(self.observations.keys())

        for time_step in dates:
            n = self.observations[time_step]
            actual_observation_keys.append(n.observation_key)
            observations.append(n.value)
            errors.append(n.std)

        dates_series = pl.Series(dates).dt.cast_time_unit("ms")

        return pl.DataFrame(
            {
                "response_key": actual_response_key,
                "observation_key": actual_observation_keys,
                "time": dates_series,
                "observations": pl.Series(observations, dtype=pl.Float32),
                "std": pl.Series(errors, dtype=pl.Float32),
            }
        )


def create_observations(
    obs_config_content: ConfContent,
    ensemble_config: EnsembleConfig,
    time_map: list[datetime] | None,
    history: HistorySource,
) -> dict[str, pl.DataFrame]:
    if not obs_config_content:
        return {}
    obs_vectors: dict[str, _GenObs | _SummaryObs] = {}
    obs_time_list: list[datetime] = []
    if ensemble_config.refcase is not None:
        obs_time_list = ensemble_config.refcase.all_dates
    elif time_map is not None:
        obs_time_list = time_map

    time_len = len(obs_time_list)
    config_errors: list[ErrorInfo] = []
    for obs_name, values in obs_config_content:
        try:
            if type(values) is HistoryValues:
                obs_vectors.update(
                    **_handle_history_observation(
                        ensemble_config,
                        values,
                        obs_name,
                        history,
                        time_len,
                    )
                )
            elif type(values) is SummaryValues:
                obs_vectors.update(
                    **_handle_summary_observation(
                        values,
                        obs_name,
                        obs_time_list,
                        bool(ensemble_config.refcase),
                    )
                )
            elif type(values) is GenObsValues:
                obs_vectors.update(
                    **_handle_general_observation(
                        ensemble_config,
                        values,
                        obs_name,
                        obs_time_list,
                        bool(ensemble_config.refcase),
                    )
                )
            else:
                config_errors.append(
                    ErrorInfo(
                        message=(
                            f"Unknown ObservationType {type(values)} for {obs_name}"
                        )
                    ).set_context(obs_name)
                )
                continue
        except ObservationConfigError as err:
            config_errors.extend(err.errors)

    if config_errors:
        raise ObservationConfigError.from_collected(config_errors)

    grouped: dict[str, list[pl.DataFrame]] = {}
    for vec in obs_vectors.values():
        if isinstance(vec, _SummaryObs):
            if "summary" not in grouped:
                grouped["summary"] = []

            grouped["summary"].append(vec.to_dataset())

        elif isinstance(vec, _GenObs):
            if "gen_data" not in grouped:
                grouped["gen_data"] = []

            grouped["gen_data"].append(vec.to_dataset())

    datasets: dict[str, pl.DataFrame] = {}

    for name, dfs in grouped.items():
        non_empty_dfs = [df for df in dfs if not df.is_empty()]
        if len(non_empty_dfs) > 0:
            ds = pl.concat(non_empty_dfs).sort("observation_key")
            if "time" in ds:
                ds = ds.sort(by="time")

            datasets[name] = ds
    return datasets


def _handle_error_mode(
    values: npt.ArrayLike,
    error_dict: ErrorValues,
) -> npt.NDArray[np.double]:
    values = np.asarray(values)
    error_mode = error_dict.error_mode
    error_min = error_dict.error_min
    error = error_dict.error
    if error_mode == "ABS":
        return np.full(values.shape, error)
    elif error_mode == "REL":
        return np.abs(values) * error
    elif error_mode == "RELMIN":
        return np.maximum(np.abs(values) * error, np.full(values.shape, error_min))
    raise ObservationConfigError(f"Unknown error mode {error_mode}", error_mode)


def _handle_history_observation(
    ensemble_config: EnsembleConfig,
    history_observation: HistoryValues,
    summary_key: str,
    history_type: HistorySource,
    time_len: int,
) -> dict[str, _SummaryObs]:
    refcase = ensemble_config.refcase
    if refcase is None:
        raise ObservationConfigError("REFCASE is required for HISTORY_OBSERVATION")

    if history_type == HistorySource.REFCASE_HISTORY:
        local_key = history_key(summary_key)
    else:
        local_key = summary_key
    if local_key is None:
        return {}
    if local_key not in refcase.keys:
        return {}
    values = refcase.values[refcase.keys.index(local_key)]
    std_dev = _handle_error_mode(values, history_observation)
    for segment_name, segment_instance in history_observation.segment:
        start = segment_instance.start
        stop = segment_instance.stop
        if start < 0:
            ConfigWarning.warn(
                f"Segment {segment_name} out of bounds."
                " Truncating start of segment to 0.",
                segment_name,
            )
            start = 0
        if stop >= time_len:
            ConfigWarning.warn(
                f"Segment {segment_name} out of bounds. Truncating"
                f" end of segment to {time_len - 1}.",
                segment_name,
            )
            stop = time_len - 1
        if start > stop:
            ConfigWarning.warn(
                f"Segment {segment_name} start after stop. Truncating"
                f" end of segment to {start}.",
                segment_name,
            )
            stop = start
        if np.size(std_dev[start:stop]) == 0:
            ConfigWarning.warn(
                f"Segment {segment_name} does not"
                " contain any time steps. The interval "
                f"[{start}, {stop}) does not intersect with steps in the"
                "time map.",
                segment_name,
            )
        std_dev[start:stop] = _handle_error_mode(
            values[start:stop],
            segment_instance,
        )
    data: dict[datetime, SummaryObservation] = {}
    for date, error, value in zip(refcase.dates, std_dev, values, strict=False):
        try:
            data[date] = SummaryObservation(
                summary_key, summary_key, value, float(error)
            )
        except ValueError as err:
            raise ObservationConfigError.with_context(str(err), summary_key) from None

    return {
        summary_key: _SummaryObs(
            summary_key,
            "summary",
            data,
        )
    }


def _get_time(
    date_dict: DateValues, start_time: datetime, context: Any = None
) -> tuple[datetime, str]:
    if date_dict.date is not None:
        return _parse_date(date_dict.date), f"DATE={date_dict.date}"
    if date_dict.days is not None:
        days = date_dict.days
        return start_time + timedelta(days=days), f"DAYS={days}"
    if date_dict.hours is not None:
        hours = date_dict.hours
        return start_time + timedelta(hours=hours), f"HOURS={hours}"
    raise ObservationConfigError.with_context("Missing time specifier", context=context)


def _parse_date(date_str: str) -> datetime:
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        try:
            date = datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError as err:
            raise ObservationConfigError.with_context(
                f"Unsupported date format {date_str}. Please use ISO date format",
                date_str,
            ) from err
        else:
            ConfigWarning.warn(
                f"Deprecated time format {date_str}."
                " Please use ISO date format YYYY-MM-DD",
                date_str,
            )
            return date


def _find_nearest(
    time_map: list[datetime],
    time: datetime,
    threshold: timedelta = DEFAULT_TIME_DELTA,
) -> int:
    nearest_index = -1
    nearest_diff = None
    for i, t in enumerate(time_map):
        diff = abs(time - t)
        if diff < threshold and (nearest_diff is None or nearest_diff > diff):
            nearest_diff = diff
            nearest_index = i
    if nearest_diff is None:
        raise IndexError(f"{time} is not in the time map")
    return nearest_index


def _get_restart(
    date_dict: DateValues,
    obs_name: str,
    time_map: list[datetime],
    has_refcase: bool,
) -> int:
    if date_dict.restart is not None:
        return date_dict.restart
    if not time_map:
        raise ObservationConfigError.with_context(
            f"Missing REFCASE or TIME_MAP for observations: {obs_name}",
            obs_name,
        )

    time, date_str = _get_time(date_dict, time_map[0], context=obs_name)

    try:
        return _find_nearest(time_map, time)
    except IndexError as err:
        raise ObservationConfigError.with_context(
            f"Could not find {time} ({date_str}) in "
            f"the time map for observations {obs_name}. "
            + (
                "The time map is set from the REFCASE keyword. Either "
                "the REFCASE has an incorrect/missing date, or the observation "
                "is given an incorrect date.)"
                if has_refcase
                else "(The time map is set from the TIME_MAP "
                "keyword. Either the time map file has an "
                "incorrect/missing date, or the observation is given an "
                "incorrect date."
            ),
            obs_name,
        ) from err


def _make_value_and_std_dev(
    observation_dict: SummaryValues,
) -> tuple[float, float]:
    value = observation_dict.value
    return (
        value,
        float(
            _handle_error_mode(
                np.array(value),
                observation_dict,
            )
        ),
    )


def _handle_summary_observation(
    summary_dict: SummaryValues,
    obs_key: str,
    time_map: list[datetime],
    has_refcase: bool,
) -> dict[str, _SummaryObs]:
    summary_key = summary_dict.key
    value, std_dev = _make_value_and_std_dev(summary_dict)

    if summary_dict.date is not None and not time_map:
        # We special case when the user has provided date in SUMMARY_OBS
        # and not REFCASE or time_map so that we don't change current behavior.
        date = _parse_date(summary_dict.date)
        restart = None
    else:
        restart = _get_restart(summary_dict, obs_key, time_map, has_refcase)
        date = time_map[restart]

    if restart == 0:
        raise ObservationConfigError.with_context(
            "It is unfortunately not possible to use summary "
            "observations from the start of the simulation. "
            f"Problem with observation {obs_key}"
            f"{' at ' + str(_get_time(summary_dict, time_map[0], obs_key)) if summary_dict.restart is None else ''}",  # noqa: E501
            obs_key,
        )
    try:
        return {
            obs_key: _SummaryObs(
                summary_key,
                "summary",
                {date: SummaryObservation(summary_key, obs_key, value, std_dev)},
            )
        }
    except ValueError as err:
        raise ObservationConfigError.with_context(str(err), obs_key) from err


def _create_gen_obs(
    scalar_value: tuple[float, float] | None = None,
    obs_file: str | None = None,
    data_index: str | None = None,
    context: Any = None,
) -> _GenObservation:
    if scalar_value is None and obs_file is None:
        raise ObservationConfigError.with_context(
            "GENERAL_OBSERVATION must contain either VALUE and ERROR or OBS_FILE",
            context=context,
        )

    if scalar_value is not None and obs_file is not None:
        raise ObservationConfigError.with_context(
            "GENERAL_OBSERVATION cannot contain both VALUE/ERROR and OBS_FILE",
            context=obs_file,
        )

    if obs_file is not None:
        try:
            file_values = np.loadtxt(obs_file, delimiter=None).ravel()
        except ValueError as err:
            raise ObservationConfigError.with_context(
                f"Failed to read OBS_FILE {obs_file}: {err}", obs_file
            ) from err
        if len(file_values) % 2 != 0:
            raise ObservationConfigError.with_context(
                "Expected even number of values in GENERAL_OBSERVATION", obs_file
            )
        values = file_values[::2]
        stds = file_values[1::2]

    else:
        assert scalar_value is not None
        obs_value, obs_std = scalar_value
        values = np.array([obs_value])
        stds = np.array([obs_std])

    if data_index is not None:
        indices = np.array([], dtype=np.int32)
        if os.path.isfile(data_index):
            indices = np.loadtxt(data_index, delimiter=None, dtype=np.int32).ravel()
        else:
            indices = np.array(sorted(rangestring_to_list(data_index)), dtype=np.int32)
    else:
        indices = np.arange(len(values), dtype=np.int32)
    std_scaling = np.full(len(values), 1.0)
    if len({len(stds), len(values), len(indices)}) != 1:
        raise ObservationConfigError.with_context(
            f"Values ({values}), error ({stds}) and "
            f"index list ({indices}) must be of equal length",
            obs_file if obs_file is not None else "",
        )
    try:
        return _GenObservation(values, stds, indices, std_scaling)
    except ValueError as err:
        raise ObservationConfigError.with_context(str(err), context) from err


def _handle_general_observation(
    ensemble_config: EnsembleConfig,
    general_observation: GenObsValues,
    obs_key: str,
    time_map: list[datetime],
    has_refcase: bool,
) -> dict[str, _GenObs]:
    response_key = general_observation.data
    if not ensemble_config.hasNodeGenData(response_key):
        ConfigWarning.warn(
            f"No GEN_DATA with name: {response_key} found - "
            f"ignoring observation {obs_key}",
            response_key,
        )
        return {}

    if all(
        getattr(general_observation, key) is None
        for key in ["restart", "date", "days", "hours"]
    ):
        # The user has not provided RESTART or DATE, this is legal
        # for GEN_DATA, so we default it to None
        restart = None
    else:
        restart = _get_restart(general_observation, obs_key, time_map, has_refcase)

    gen_data_config = ensemble_config.response_configs.get("gen_data", None)
    assert isinstance(gen_data_config, GenDataConfig)
    if response_key not in gen_data_config.keys:
        ConfigWarning.warn(
            f"Observation {obs_key} on GEN_DATA key {response_key}, but GEN_DATA"
            f" key {response_key} is non-existing"
        )
        return {}

    _, report_steps = gen_data_config.get_args_for_key(response_key)

    response_report_steps = [] if report_steps is None else report_steps
    if (restart is None and response_report_steps) or (
        restart is not None and restart not in response_report_steps
    ):
        ConfigWarning.warn(
            f"The GEN_DATA node:{response_key} is not configured to load from"
            f" report step:{restart} for the observation:{obs_key}"
            " - The observation will be ignored",
            response_key,
        )
        return {}

    restart = 0 if restart is None else restart
    index_list = general_observation.index_list
    index_file = general_observation.index_file
    if index_list is not None and index_file is not None:
        raise ObservationConfigError.with_context(
            f"GENERAL_OBSERVATION {obs_key} has both INDEX_FILE and INDEX_LIST.",
            obs_key,
        )
    indices = index_list if index_list is not None else index_file
    return {
        obs_key: _GenObs(
            obs_key,
            response_key,
            {
                restart: _create_gen_obs(
                    (
                        (
                            general_observation.value,
                            general_observation.error,
                        )
                        if general_observation.value is not None
                        and general_observation.error is not None
                        else None
                    ),
                    general_observation.obs_file,
                    indices,
                    obs_key,
                ),
            },
        )
    }
