from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, assert_never

import numpy as np
import polars as pl

from ert.summary_key_type import history_key
from ert.validation import rangestring_to_list

from ._observations import (
    GeneralObservation,
    HistoryObservation,
    Observation,
    ObservationDate,
    ObservationError,
    SummaryObservation,
)
from .ensemble_config import EnsembleConfig
from .gen_data_config import GenDataConfig
from .parsing import (
    ConfigWarning,
    ErrorInfo,
    HistorySource,
    ObservationConfigError,
)

if TYPE_CHECKING:
    import numpy.typing as npt


DEFAULT_TIME_DELTA = timedelta(seconds=30)


def create_observation_dataframes(
    observations: Sequence[Observation],
    ensemble_config: EnsembleConfig,
    time_map: list[datetime] | None,
    history: HistorySource,
) -> dict[str, pl.DataFrame]:
    if not observations:
        return {}
    obs_time_list: list[datetime] = []
    if ensemble_config.refcase is not None:
        obs_time_list = ensemble_config.refcase.all_dates
    elif time_map is not None:
        obs_time_list = time_map

    time_len = len(obs_time_list)
    config_errors: list[ErrorInfo] = []
    grouped: dict[str, list[pl.DataFrame]] = defaultdict(list)
    for obs in observations:
        obs_name = obs.name
        try:
            match obs:
                case HistoryObservation():
                    grouped["summary"].append(
                        _handle_history_observation(
                            ensemble_config,
                            obs,
                            obs_name,
                            history,
                            time_len,
                        )
                    )
                case SummaryObservation():
                    grouped["summary"].append(
                        _handle_summary_observation(
                            obs,
                            obs_name,
                            obs_time_list,
                            bool(ensemble_config.refcase),
                        )
                    )
                case GeneralObservation():
                    grouped["gen_data"].append(
                        _handle_general_observation(
                            ensemble_config,
                            obs,
                            obs_name,
                            obs_time_list,
                            bool(ensemble_config.refcase),
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


def _handle_error_mode(
    values: npt.ArrayLike,
    error_dict: ObservationError,
) -> npt.NDArray[np.double]:
    values = np.asarray(values)
    error_mode = error_dict.error_mode
    error_min = error_dict.error_min
    error = error_dict.error
    match error_mode:
        case "ABS":
            return np.full(values.shape, error)
        case "REL":
            return np.abs(values) * error
        case "RELMIN":
            return np.maximum(np.abs(values) * error, np.full(values.shape, error_min))
        case default:
            assert_never(default)


def _handle_history_observation(
    ensemble_config: EnsembleConfig,
    history_observation: HistoryObservation,
    summary_key: str,
    history_type: HistorySource,
    time_len: int,
) -> pl.DataFrame:
    refcase = ensemble_config.refcase
    if refcase is None:
        raise ObservationConfigError.with_context(
            "REFCASE is required for HISTORY_OBSERVATION", summary_key
        )

    if history_type == HistorySource.REFCASE_HISTORY:
        local_key = history_key(summary_key)
    else:
        local_key = summary_key
    if local_key not in refcase.keys:
        raise ObservationConfigError.with_context(
            f"Key {local_key!r} is not present in refcase", summary_key
        )
    values = refcase.values[refcase.keys.index(local_key)]
    std_dev = _handle_error_mode(values, history_observation)
    for segment in history_observation.segments:
        start = segment.start
        stop = segment.stop
        if start < 0:
            ConfigWarning.warn(
                f"Segment {segment.name} out of bounds."
                " Truncating start of segment to 0.",
                segment.name,
            )
            start = 0
        if stop >= time_len:
            ConfigWarning.warn(
                f"Segment {segment.name} out of bounds. Truncating"
                f" end of segment to {time_len - 1}.",
                segment.name,
            )
            stop = time_len - 1
        if start > stop:
            ConfigWarning.warn(
                f"Segment {segment.name} start after stop. Truncating"
                f" end of segment to {start}.",
                segment.name,
            )
            stop = start
        if np.size(std_dev[start:stop]) == 0:
            ConfigWarning.warn(
                f"Segment {segment.name} does not"
                " contain any time steps. The interval "
                f"[{start}, {stop}) does not intersect with steps in the"
                "time map.",
                segment.name,
            )
        std_dev[start:stop] = _handle_error_mode(values[start:stop], segment)
    dates_series = pl.Series(refcase.dates).dt.cast_time_unit("ms")
    if (std_dev <= 0).any():
        raise ObservationConfigError.with_context(
            "Observation uncertainty must be strictly > 0", summary_key
        ) from None

    return pl.DataFrame(
        {
            "response_key": summary_key,
            "observation_key": summary_key,
            "time": dates_series,
            "observations": pl.Series(values, dtype=pl.Float32),
            "std": pl.Series(std_dev, dtype=pl.Float32),
        }
    )


def _get_time(
    date_dict: ObservationDate, start_time: datetime, context: Any = None
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
    date_dict: ObservationDate,
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


def _handle_summary_observation(
    summary_dict: SummaryObservation,
    obs_key: str,
    time_map: list[datetime],
    has_refcase: bool,
) -> pl.DataFrame:
    summary_key = summary_dict.key
    value = summary_dict.value
    std_dev = float(_handle_error_mode(np.array(value), summary_dict))

    if summary_dict.restart and not (time_map or has_refcase):
        raise ObservationConfigError.with_context(
            "Keyword 'RESTART' requires either TIME_MAP or REFCASE", context=obs_key
        )

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
    if std_dev <= 0:
        raise ObservationConfigError.with_context(
            "Observation uncertainty must be strictly > 0", summary_key
        ) from None

    return pl.DataFrame(
        {
            "response_key": [summary_key],
            "observation_key": [obs_key],
            "time": pl.Series([date]).dt.cast_time_unit("ms"),
            "observations": pl.Series([value], dtype=pl.Float32),
            "std": pl.Series([std_dev], dtype=pl.Float32),
        }
    )


def _handle_general_observation(
    ensemble_config: EnsembleConfig,
    general_observation: GeneralObservation,
    obs_key: str,
    time_map: list[datetime],
    has_refcase: bool,
) -> pl.DataFrame:
    response_key = general_observation.data

    if all(
        getattr(general_observation, key) is None
        for key in ["restart", "date", "days", "hours"]
    ):
        # The user has not provided RESTART or DATE, this is legal
        # for GEN_DATA, so we default it to None
        restart = None
    else:
        restart = _get_restart(general_observation, obs_key, time_map, has_refcase)

    if (
        (gen_data_config := ensemble_config.response_configs.get("gen_data", None))
        is None
    ) or response_key not in gen_data_config.keys:
        raise ObservationConfigError.with_context(
            f"Problem with GENERAL_OBSERVATION {obs_key}:"
            f" No GEN_DATA with name {response_key!r} found",
            response_key,
        )
    assert isinstance(gen_data_config, GenDataConfig)

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
        general_observation.value is None
        and general_observation.error is None
        and general_observation.obs_file is None
    ):
        raise ObservationConfigError.with_context(
            "GENERAL_OBSERVATION must contain either VALUE and ERROR or OBS_FILE",
            context=obs_key,
        )

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
        assert (
            general_observation.value is not None
            and general_observation.error is not None
        )
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
            general_observation.obs_file
            if general_observation.obs_file is not None
            else "",
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
        }
    )
