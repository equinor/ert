from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from ert.config.parsing import ConfigWarning
from ert.config.parsing.observations_parser import (
    DateValues,
    ErrorValues,
    ObservationConfigError,
    SummaryValues,
)

DEFAULT_TIME_DELTA = timedelta(seconds=30)


def get_time(date_dict: DateValues, start_time: datetime) -> Tuple[datetime, str]:
    if date_dict.date is not None:
        date_str = date_dict.date
        try:
            return datetime.fromisoformat(date_str), f"DATE={date_str}"
        except ValueError:
            try:
                date = datetime.strptime(date_str, "%d/%m/%Y")
                ConfigWarning.ert_context_warn(
                    f"Deprecated time format {date_str}."
                    " Please use ISO date format YYYY-MM-DD",
                    date_str,
                )
                return date, f"DATE={date_str}"
            except ValueError as err:
                raise ObservationConfigError.with_context(
                    f"Unsupported date format {date_str}."
                    " Please use ISO date format",
                    date_str,
                ) from err

    if date_dict.days is not None:
        days = date_dict.days
        return start_time + timedelta(days=days), f"DAYS={days}"
    if date_dict.hours is not None:
        hours = date_dict.hours
        return start_time + timedelta(hours=hours), f"HOURS={hours}"
    raise ValueError("Missing time specifier")


def _find_nearest(
    time_map: List[datetime],
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


def get_restart(
    date_dict: DateValues,
    obs_name: str,
    time_map: List[datetime],
    has_refcase: bool,
) -> int:
    if date_dict.restart is not None:
        return date_dict.restart
    if not time_map:
        raise ObservationConfigError.with_context(
            f"Missing REFCASE or TIME_MAP for observations: {obs_name}",
            obs_name,
        )

    try:
        time, date_str = get_time(date_dict, time_map[0])
    except ObservationConfigError:
        raise
    except ValueError as err:
        raise ObservationConfigError.with_context(
            f"Failed to parse date of {obs_name}", obs_name
        ) from err

    try:
        return _find_nearest(time_map, time)
    except IndexError as err:
        raise ObservationConfigError.with_context(
            f"Could not find {time} ({date_str}) in "
            f"the time map for observations {obs_name}"
            + (
                "The time map is set from the REFCASE keyword. Either "
                "the REFCASE has an incorrect/missing date, or the observation "
                "is given an incorrect date.)"
                if has_refcase
                else " (The time map is set from the TIME_MAP "
                "keyword. Either the time map file has an "
                "incorrect/missing date, or the  observation is given an "
                "incorrect date."
            ),
            obs_name,
        ) from err


def make_value_and_std_dev(
    observation_dict: SummaryValues,
) -> Tuple[float, float]:
    value = observation_dict.value
    return (
        value,
        float(
            handle_error_mode(
                np.array(value),
                observation_dict,
            )
        ),
    )


def handle_error_mode(
    values: "npt.ArrayLike",
    error_dict: ErrorValues,
) -> "npt.NDArray[np.double]":
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
