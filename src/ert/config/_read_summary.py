"""
Functions for reading reservoir simulator summary files. See
[OPM flow 2024.04 manual Appendix F.9](https://opm-project.org/?page_id=955)
for specification of the file format
"""

from __future__ import annotations

import fnmatch
import re
import warnings
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from enum import Enum, auto

import numpy as np
import numpy.typing as npt
from resfo_utilities import (
    InvalidSummaryError,
    InvalidSummaryKeyError,
    SummaryReader,
    make_summary_key,
)

from .response_config import InvalidResponseFile


def read_summary(
    summary_basename: str, select_keys: Sequence[str]
) -> tuple[datetime, list[str], Sequence[datetime], npt.NDArray[np.float32]]:
    """Reads the timeseries for the selected keys from summary files with the
    given basename.

    Called as read_summary("data/CASE", ["FOP*"]) it will read from files
    data/CASE.UNSMRY & data/CASE.SMSPEC and return the tuple (start_date, keys,
    times, values) where

    * start_date is the start_date for the simulation
    * keys is list of keys that matched the selection "FOP*" ("*" means wildcard)
    * times is the x-axis for the time series
    * values is an array of dimensions len(keys) * len(times) with y-axis
    values.

    Note that if formatted files are present (data/CASE.FUNSMRY and
    data/CASE.FSMSPEC) then those will be read from. It is also possible to
    give the simulator input file, read_summary("data/CASE.DATA", ["FOP*"]),
    and it will then read from the corresponding summary files data/CASE.UNSMRY &
    data/CASE.SMSPEC.

    """
    if summary_basename.lower().endswith(".data"):
        # For backwards compatability, it is
        # allowed to give REFCASE and ECLBASE both
        # with and without .DATA extensions
        summary_basename = summary_basename[:-5]
    summary = SummaryReader(case_path=summary_basename)
    try:
        date_index, start_date, date_units, keys, indices = _read_spec(
            summary, select_keys
        )
        fetched, time_map = _read_summary(
            summary, start_date, date_units, indices, date_index
        )
    except InvalidSummaryError as err:
        raise InvalidResponseFile(
            f"Failed to read summary files {summary_basename}: {err}"
        ) from err
    return (start_date, keys, time_map, fetched)


__all__ = ["read_summary"]


class DateUnit(Enum):
    HOURS = auto()
    DAYS = auto()

    def make_delta(self, val: float) -> timedelta:
        if self == DateUnit.HOURS:
            return timedelta(hours=val)
        if self == DateUnit.DAYS:
            return timedelta(days=val)
        raise InvalidResponseFile(f"Unknown date unit {val}")


def _fetch_keys_to_matcher(fetch_keys: Sequence[str]) -> Callable[[str], bool]:
    """
    Transform the list of keys (with * used as repeated wildcard) into
    a matcher.

    >>> match = _fetch_keys_to_matcher([""])
    >>> match("FOPR")
    False

    >>> match = _fetch_keys_to_matcher(["*"])
    >>> match("FOPR"), match("FO*")
    (True, True)


    >>> match = _fetch_keys_to_matcher(["F*PR"])
    >>> match("WOPR"), match("FOPR"), match("FGPR"), match("SOIL")
    (False, True, True, False)

    >>> match = _fetch_keys_to_matcher(["WGOR:*"])
    >>> match("FOPR"), match("WGOR:OP1"), match("WGOR:OP2"), match("WGOR")
    (False, True, True, False)

    >>> match = _fetch_keys_to_matcher(["FOPR", "FGPR"])
    >>> match("FOPR"), match("FGPR"), match("WGOR:OP2"), match("WGOR")
    (True, True, False, False)
    """
    if not fetch_keys:
        return lambda _: False
    regex = re.compile("|".join(fnmatch.translate(key) for key in fetch_keys))
    return lambda s: regex.fullmatch(s) is not None


def _read_spec(
    summary: SummaryReader, fetch_keys: Sequence[str]
) -> tuple[int, datetime, DateUnit, list[str], npt.NDArray[np.int64]]:
    date = summary.start_date
    dims = summary.dimensions
    if dims is not None:
        nx, ny = dims[0:2]
    else:
        nx, ny = None, None

    keywords = summary.summary_keywords

    if date is None:
        raise InvalidResponseFile(
            f"Keyword STARTDAT missing in {summary.smspec_filename}"
        )

    indices: list[int] = []
    keys: list[str] = []
    index_mapping: dict[str, int] = {}
    date_index = None
    date_unit_str = None

    should_load_key = _fetch_keys_to_matcher(fetch_keys)

    for i, kw in enumerate(keywords):
        try:
            key = make_summary_key(
                kw.summary_variable,
                kw.number,
                kw.name,
                nx,
                ny,
                kw.lgr_name,
                kw.li,
                kw.lj,
                kw.lk,
            )
            if kw.summary_variable == "TIME":
                date_index = i
                date_unit_str = kw.unit
        except InvalidSummaryKeyError as err:
            warnings.warn(
                f"Found {err} in summary specification, key not loaded", stacklevel=2
            )
            continue

        if should_load_key(key):
            if key in index_mapping:
                # only keep the index of the last occurrence of a key
                # this is done for backwards compatability
                # and to have unique keys
                indices[index_mapping[key]] = i
            else:
                index_mapping[key] = len(indices)
                indices.append(i)
                keys.append(key)

    keys_array = np.array(keys)
    rearranged = keys_array.argsort()
    keys_array = keys_array[rearranged]

    indices_array = np.array(indices, dtype=np.int64)[rearranged]

    if date_index is None:
        raise InvalidResponseFile(
            f"KEYWORDS did not contain TIME in {summary.smspec_filename}"
        )
    if date_unit_str is None:
        raise InvalidResponseFile(f"Unit missing for TIME in {summary.smspec_filename}")

    try:
        date_unit = DateUnit[date_unit_str]
    except KeyError:
        raise InvalidResponseFile(
            f"Unknown date unit in {summary.smspec_filename}: {date_unit_str}"
        ) from None

    return (
        date_index,
        date,
        date_unit,
        list(keys_array),
        indices_array,
    )


def _round_to_seconds(dt: datetime) -> datetime:
    """
    >>> _round_to_seconds(datetime(2000, 1, 1, 1, 0, 1, 1))
    datetime.datetime(2000, 1, 1, 1, 0, 1)
    >>> _round_to_seconds(datetime(2000, 1, 1, 1, 0, 1, 500001))
    datetime.datetime(2000, 1, 1, 1, 0, 2)
    >>> _round_to_seconds(datetime(2000, 1, 1, 1, 0, 1, 499999))
    datetime.datetime(2000, 1, 1, 1, 0, 1)
    """
    extra_sec = round(dt.microsecond / 10**6)
    return dt.replace(microsecond=0) + timedelta(seconds=extra_sec)


def _read_summary(
    summary: SummaryReader,
    start_date: datetime,
    unit: DateUnit,
    indices: npt.NDArray[np.int64],
    date_index: int,
) -> tuple[npt.NDArray[np.float32], list[datetime]]:
    values: list[npt.NDArray[np.float32]] = []
    dates: list[datetime] = []
    try:
        for v in summary.values(report_step_only=True):
            values.append(v[indices])
            dates.append(
                _round_to_seconds(start_date + unit.make_delta(float(v[date_index]))),
            )
        return np.array(values, dtype=np.float32).T, dates
    except ValueError as e:
        raise InvalidResponseFile(f"Unable to read summary data from {summary}") from e
