"""
Functions for reading reservoir simulator summary files. See
[OPM flow 2024.04 manual Appendix F.9](https://opm-project.org/?page_id=955)
for specification of the file format
"""

from __future__ import annotations

import fnmatch
import os
import os.path
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
import resfo
from pydantic import PositiveInt

from ert.summary_key_type import SummaryKeyType

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
    summary, spec = _get_summary_filenames(summary_basename)
    try:
        date_index, start_date, date_units, keys, indices = _read_spec(
            spec, select_keys
        )
        fetched, time_map = _read_summary(
            summary, start_date, date_units, indices, date_index
        )
    except resfo.ResfoParsingError as err:
        raise InvalidResponseFile(
            f"Failed to read summary file {summary_basename}: {err}"
        ) from err
    return (start_date, keys, time_map, fetched)


def make_summary_key(
    keyword: str,
    number: int | None = None,
    name: str | None = None,
    nx: int | None = None,
    ny: int | None = None,
    lgr_name: str | None = None,
    li: int | None = None,
    lj: int | None = None,
    lk: int | None = None,
) -> str:
    """Converts values found in the summary file to the summary_key format.

    Ert and some other applications use a colon separated format to specify
    summary parameters. For instance:

    >>> make_summary_key(keyword="WOPR", name="WELL1")
    'WOPR:WELL1'
    >>> make_summary_key(keyword="BOPR", number=4, nx=2, ny=2)
    'BOPR:2,2,1'
    """
    try:
        sum_type = SummaryKeyType.from_variable(keyword)
    except Exception as err:
        raise InvalidResponseFile(
            f"Could not read summary keyword '{keyword}': {err}"
        ) from err

    match sum_type:
        case SummaryKeyType.FIELD | SummaryKeyType.OTHER:
            return keyword
        case SummaryKeyType.REGION | SummaryKeyType.AQUIFER:
            return f"{keyword}:{number}"
        case SummaryKeyType.BLOCK:
            nx, ny = _check_if_missing("block", "dimens", nx, ny)
            (number,) = _check_if_missing("block", "nums", number)
            i, j, k = _cell_index(number - 1, nx, ny)
            return f"{keyword}:{i},{j},{k}"
        case SummaryKeyType.GROUP | SummaryKeyType.WELL:
            return f"{keyword}:{name}"
        case SummaryKeyType.SEGMENT:
            return f"{keyword}:{name}:{number}"
        case SummaryKeyType.COMPLETION:
            nx, ny = _check_if_missing("completion", "dimens", nx, ny)
            (number,) = _check_if_missing("completion", "nums", number)
            i, j, k = _cell_index(number - 1, nx, ny)
            return f"{keyword}:{name}:{i},{j},{k}"
        case SummaryKeyType.INTER_REGION:
            (number,) = _check_if_missing("inter region", "nums", number)
            r1 = number % 32768
            r2 = ((number - r1) // 32768) - 10
            return f"{keyword}:{r1}-{r2}"
        case SummaryKeyType.LOCAL_WELL:
            (name,) = _check_if_missing("local well", "WGNAMES", name)
            (lgr_name,) = _check_if_missing("local well", "LGRS", lgr_name)
            return f"{keyword}:{lgr_name}:{name}"
        case SummaryKeyType.LOCAL_BLOCK:
            li, lj, lk = _check_if_missing("local block", "NUMLX", li, lj, lk)
            (lgr_name,) = _check_if_missing("local block", "LGRS", lgr_name)
            return f"{keyword}:{lgr_name}:{li},{lj},{lk}"
        case SummaryKeyType.LOCAL_COMPLETION:
            li, lj, lk = _check_if_missing("local completion", "NUMLX", li, lj, lk)
            (name,) = _check_if_missing("local completion", "WGNAMES", name)
            (lgr_name,) = _check_if_missing("local completion", "LGRS", lgr_name)
            return f"{keyword}:{lgr_name}:{name}:{li},{lj},{lk}"
        case SummaryKeyType.NETWORK:
            (name,) = _check_if_missing("network", "WGNAMES", name)
            return f"{keyword}:{name}"


__all__ = ["make_summary_key", "read_summary"]


def _cell_index(
    array_index: int, nx: PositiveInt, ny: PositiveInt
) -> tuple[int, int, int]:
    k = array_index // (nx * ny)
    array_index -= k * (nx * ny)
    j = array_index // nx
    array_index -= j * nx

    return array_index + 1, j + 1, k + 1


T = TypeVar("T")


def _check_if_missing(
    keyword_name: str, missing_key: str, *test_vars: T | None
) -> list[T]:
    if any(v is None for v in test_vars):
        raise InvalidResponseFile(
            f"Found {keyword_name} keyword in summary "
            f"specification without {missing_key} keyword"
        )
    return test_vars  # type: ignore


class DateUnit(Enum):
    HOURS = auto()
    DAYS = auto()

    def make_delta(self, val: float) -> timedelta:
        if self == DateUnit.HOURS:
            return timedelta(hours=val)
        if self == DateUnit.DAYS:
            return timedelta(days=val)
        raise InvalidResponseFile(f"Unknown date unit {val}")


def _is_base_with_extension(base: str, path: str, exts: list[str]) -> bool:
    """
    >>> _is_base_with_extension("ECLBASE", "ECLBASE.SMSPEC", ["smspec"])
    True
    >>> _is_base_with_extension("ECLBASE", "BASE.SMSPEC", ["smspec"])
    False
    >>> _is_base_with_extension("ECLBASE", "BASE.FUNSMRY", ["smspec"])
    False
    >>> _is_base_with_extension("ECLBASE", "ECLBASE.smspec", ["smspec"])
    True
    >>> _is_base_with_extension("ECLBASE.tar.gz", "ECLBASE.tar.gz.smspec", ["smspec"])
    True
    """
    if "." not in path:
        return False
    splitted = path.split(".")
    return ".".join(splitted[0:-1]) == base and splitted[-1].lower() in exts


def _is_unsmry(base: str, path: str) -> bool:
    return _is_base_with_extension(base, path, ["unsmry", "funsmry"])


def _is_smspec(base: str, path: str) -> bool:
    return _is_base_with_extension(base, path, ["smspec", "fsmspec"])


def _find_file_matching(
    kind: str, case: str, predicate: Callable[[str, str], bool]
) -> str:
    directory, base = os.path.split(case)
    candidates = list(
        filter(lambda x: predicate(base, x), os.listdir(directory or "."))
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find any {kind} matching case path {case}")
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Ambiguous reference to {kind} in {case}, could be any of {candidates}"
        )
    return os.path.join(directory, candidates[0])


def _get_summary_filenames(filepath: str) -> tuple[str, str]:
    if filepath.lower().endswith(".data"):
        # For backwards compatability, it is
        # allowed to give REFCASE and ECLBASE both
        # with and without .DATA extensions
        filepath = filepath[:-5]
    summary = _find_file_matching("unified summary file", filepath, _is_unsmry)
    spec = _find_file_matching("smspec file", filepath, _is_smspec)
    return summary, spec


def _key2str(key: bytes | str) -> str:
    ret = key.decode() if isinstance(key, bytes) else key
    assert isinstance(ret, str)
    return ret.strip()


def _check_vals(
    kw: str, spec: str, vals: npt.NDArray[Any] | resfo.MESS
) -> npt.NDArray[Any]:
    if vals is resfo.MESS or isinstance(vals, resfo.MESS):
        raise InvalidResponseFile(f"{kw.strip()} in {spec} has incorrect type MESS")
    return vals


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
    spec: str, fetch_keys: Sequence[str]
) -> tuple[int, datetime, DateUnit, list[str], npt.NDArray[np.int64]]:
    date = None
    n = None
    nx = None
    ny = None
    wgnames = None

    arrays: dict[str, npt.NDArray[Any] | None] = dict.fromkeys(
        [
            "NUMS    ",
            "KEYWORDS",
            "NUMLX   ",
            "NUMLY   ",
            "NUMLZ   ",
            "LGRS    ",
            "UNITS   ",
        ],
        None,
    )
    if spec.lower().endswith("fsmspec"):
        mode = "rt"
        assumed_format = resfo.Format.FORMATTED
    else:
        mode = "rb"
        assumed_format = resfo.Format.UNFORMATTED

    with open(spec, mode) as fp:
        for entry in resfo.lazy_read(fp, assumed_format):
            if all(p is not None for p in [date, n, nx, ny, *arrays.values()]):
                break
            kw = entry.read_keyword()
            if kw in arrays:
                arrays[kw] = _check_vals(kw, spec, entry.read_array())
            if kw in {"WGNAMES ", "NAMES   "}:
                wgnames = _check_vals(kw, spec, entry.read_array())
            if kw == "DIMENS  ":
                vals = _check_vals(kw, spec, entry.read_array())
                size = len(vals)
                n = vals[0] if size > 0 else None
                nx = vals[1] if size > 1 else None
                ny = vals[2] if size > 2 else None
            if kw == "STARTDAT":
                vals = _check_vals(kw, spec, entry.read_array())
                size = len(vals)
                day = vals[0] if size > 0 else 0
                month = vals[1] if size > 1 else 0
                year = vals[2] if size > 2 else 0
                hour = vals[3] if size > 3 else 0
                minute = vals[4] if size > 4 else 0
                microsecond = vals[5] if size > 5 else 0
                try:
                    date = datetime(
                        day=day,
                        month=month,
                        year=year,
                        hour=hour,
                        minute=minute,
                        second=microsecond // 10**6,
                        # Due to https://github.com/equinor/ert/issues/6952
                        # microseconds have to be ignored to avoid overflow
                        # in netcdf3 files
                        # microsecond=self.micro_seconds % 10**6,
                    )
                except Exception as err:
                    raise InvalidResponseFile(
                        f"SMSPEC {spec} contains invalid STARTDAT: {err}"
                    ) from err
    keywords = arrays["KEYWORDS"]
    nums = arrays["NUMS    "]
    numlx = arrays["NUMLX   "]
    numly = arrays["NUMLY   "]
    numlz = arrays["NUMLZ   "]
    lgr_names = arrays["LGRS    "]

    if date is None:
        raise InvalidResponseFile(f"Keyword startdat missing in {spec}")
    if keywords is None:
        raise InvalidResponseFile(f"Keywords missing in {spec}")
    if n is None:
        n = len(keywords)

    indices: list[int] = []
    keys: list[str] = []
    index_mapping: dict[str, int] = {}
    date_index = None

    should_load_key = _fetch_keys_to_matcher(fetch_keys)

    def optional_get(arr: npt.NDArray[Any] | None, idx: int) -> Any:
        if arr is None:
            return None
        if len(arr) <= idx:
            return None
        return arr[idx]

    for i in range(n):
        keyword = _key2str(keywords[i])
        if keyword == "TIME":
            date_index = i

        name = optional_get(wgnames, i)
        if name is not None:
            name = _key2str(name)
        num = optional_get(nums, i)
        lgr_name = optional_get(lgr_names, i)
        if lgr_name is not None:
            lgr_name = _key2str(lgr_name)
        li = optional_get(numlx, i)
        lj = optional_get(numly, i)
        lk = optional_get(numlz, i)

        key = make_summary_key(keyword, num, name, nx, ny, lgr_name, li, lj, lk)
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

    units = arrays["UNITS   "]
    if units is None:
        raise InvalidResponseFile(f"Keyword units missing in {spec}")
    if date_index is None:
        raise InvalidResponseFile(f"KEYWORDS did not contain TIME in {spec}")
    if date_index >= len(units):
        raise InvalidResponseFile(f"Unit missing for TIME in {spec}")

    unit_key = _key2str(units[date_index])
    try:
        date_unit = DateUnit[unit_key]
    except KeyError:
        raise InvalidResponseFile(f"Unknown date unit in {spec}: {unit_key}") from None

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
    summary: str,
    start_date: datetime,
    unit: DateUnit,
    indices: npt.NDArray[np.int64],
    date_index: int,
) -> tuple[npt.NDArray[np.float32], list[datetime]]:
    if summary.lower().endswith("funsmry"):
        mode = "rt"
        assumed_format = resfo.Format.FORMATTED
    else:
        mode = "rb"
        assumed_format = resfo.Format.UNFORMATTED

    last_params = None
    values: list[npt.NDArray[np.float32]] = []
    dates: list[datetime] = []

    def read_params() -> None:
        nonlocal last_params, values
        if last_params is not None:
            vals = _check_vals("PARAMS", summary, last_params.read_array())
            values.append(vals[indices])

            dates.append(
                _round_to_seconds(
                    start_date + unit.make_delta(float(vals[date_index]))
                ),
            )
            # Due to https://github.com/equinor/ert/issues/6952
            # times have to be rounded to whole seconds to avoid overflow
            # in netcdf3 files
            # dates.append(start_date + unit.make_delta(float(vals[date_index])))
            last_params = None

    try:
        with open(summary, mode) as fp:
            for entry in resfo.lazy_read(fp, assumed_format):
                kw = entry.read_keyword()
                if kw == "PARAMS  ":
                    last_params = entry
                if kw == "SEQHDR  ":
                    read_params()
            read_params()
    except ValueError as e:
        raise InvalidResponseFile(f"Unable to read summary data from {summary}") from e
    return np.array(values, dtype=np.float32).T, dates
