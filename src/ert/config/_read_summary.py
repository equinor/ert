from __future__ import annotations

import os
import os.path
import re
from datetime import datetime, timedelta
from enum import Enum, auto
from fnmatch import fnmatch
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import resfo
from pydantic import PositiveInt

SPECIAL_KEYWORDS = [
    "NEWTON",
    "NAIMFRAC",
    "NLINEARS",
    "NLINSMIN",
    "NLINSMAX",
    "ELAPSED",
    "MAXDPR",
    "MAXDSO",
    "MAXDSG",
    "MAXDSW",
    "STEPTYPE",
    "WNEWTON",
]


class _SummaryType(Enum):
    AQUIFER = auto()
    BLOCK = auto()
    COMPLETION = auto()
    FIELD = auto()
    GROUP = auto()
    LOCAL_BLOCK = auto()
    LOCAL_COMPLETION = auto()
    LOCAL_WELL = auto()
    NETWORK = auto()
    SEGMENT = auto()
    WELL = auto()
    REGION = auto()
    INTER_REGION = auto()
    OTHER = auto()

    @classmethod
    def from_keyword(cls, summary_keyword: str) -> _SummaryType:
        KEYWORD_TYPE_MAPPING = {
            "A": cls.AQUIFER,
            "B": cls.BLOCK,
            "C": cls.COMPLETION,
            "F": cls.FIELD,
            "G": cls.GROUP,
            "LB": cls.LOCAL_BLOCK,
            "LC": cls.LOCAL_COMPLETION,
            "LW": cls.LOCAL_WELL,
            "N": cls.NETWORK,
            "S": cls.SEGMENT,
            "W": cls.WELL,
        }
        if summary_keyword == "":
            raise ValueError("Got empty summary_keyword")
        if any(special in summary_keyword for special in SPECIAL_KEYWORDS):
            return cls.OTHER
        if summary_keyword[0] in KEYWORD_TYPE_MAPPING:
            return KEYWORD_TYPE_MAPPING[summary_keyword[0]]
        if summary_keyword[0:2] in KEYWORD_TYPE_MAPPING:
            return KEYWORD_TYPE_MAPPING[summary_keyword[0:2]]
        if summary_keyword == "RORFR":
            return cls.REGION

        if any(
            re.match(pattern, summary_keyword)
            for pattern in [r"R.FT.*", r"R..FT.*", r"R.FR.*", r"R..FR.*", r"R.F"]
        ):
            return cls.INTER_REGION
        if summary_keyword[0] == "R":
            return cls.REGION

        return cls.OTHER


def _cell_index(
    array_index: int, nx: PositiveInt, ny: PositiveInt
) -> Tuple[int, int, int]:
    k = array_index // (nx * ny)
    array_index -= k * (nx * ny)
    j = array_index // nx
    array_index -= j * nx

    return array_index + 1, j + 1, k + 1


def _check_if_missing(
    keyword_name: str, missing_key: str, *test_vars: Optional[int]
) -> List[int]:
    if any(v is None for v in test_vars):
        raise ValueError(
            f"Found {keyword_name} keyword in summary "
            f"specification without {missing_key} keyword"
        )
    return test_vars  # type: ignore


def make_summary_key(
    keyword: str,
    number: Optional[int] = None,
    name: Optional[str] = None,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    lgr_name: Optional[str] = None,
    li: Optional[int] = None,
    lj: Optional[int] = None,
    lk: Optional[int] = None,
) -> Optional[str]:
    sum_type = _SummaryType.from_keyword(keyword)
    if sum_type in [
        _SummaryType.FIELD,
        _SummaryType.OTHER,
    ]:
        return keyword
    if sum_type in [
        _SummaryType.REGION,
        _SummaryType.AQUIFER,
    ]:
        return f"{keyword}:{number}"
    if sum_type == _SummaryType.BLOCK:
        nx, ny = _check_if_missing("block", "dimens", nx, ny)
        (number,) = _check_if_missing("block", "nums", number)
        i, j, k = _cell_index(number - 1, nx, ny)
        return f"{keyword}:{i},{j},{k}"
    if sum_type in [
        _SummaryType.GROUP,
        _SummaryType.WELL,
    ]:
        return f"{keyword}:{name}"
    if sum_type == _SummaryType.SEGMENT:
        return f"{keyword}:{name}:{number}"
    if sum_type == _SummaryType.COMPLETION:
        nx, ny = _check_if_missing("completion", "dimens", nx, ny)
        (number,) = _check_if_missing("completion", "nums", number)
        i, j, k = _cell_index(number - 1, nx, ny)
        return f"{keyword}:{name}:{i},{j},{k}"
    if sum_type == _SummaryType.INTER_REGION:
        (number,) = _check_if_missing("inter region", "nums", number)
        r1 = number % 32768
        r2 = ((number - r1) // 32768) - 10
        return f"{keyword}:{r1}-{r2}"
    if sum_type == _SummaryType.LOCAL_WELL:
        return f"{keyword}:{lgr_name}:{name}"
    if sum_type == _SummaryType.LOCAL_BLOCK:
        return f"{keyword}:{lgr_name}:{li},{lj},{lk}"
    if sum_type == _SummaryType.LOCAL_COMPLETION:
        nx, ny = _check_if_missing("local completion", "dimens", nx, ny)
        (number,) = _check_if_missing("local completion", "nums", number)
        i, j, k = _cell_index(number - 1, nx, ny)
        return f"{keyword}:{lgr_name}:{name}:{li},{lj},{lk}"
    if sum_type == _SummaryType.NETWORK:
        # This is consistent with resinsight but
        # has a bug in resdata
        # https://github.com/equinor/resdata/issues/943
        return keyword
    raise ValueError(f"Unexpected keyword type: {sum_type}")


class DateUnit(Enum):
    HOURS = auto()
    DAYS = auto()

    def make_delta(self, val: float) -> timedelta:
        if self == DateUnit.HOURS:
            return timedelta(hours=val)
        if self == DateUnit.DAYS:
            return timedelta(days=val)
        raise ValueError(f"Unknown date unit {val}")


def _is_unsmry(base: str, path: str) -> bool:
    if "." not in path:
        return False
    splitted = path.split(".")
    return splitted[-2].endswith(base) and splitted[-1].lower() in ["unsmry", "funsmry"]


def _is_smspec(base: str, path: str) -> bool:
    if "." not in path:
        return False
    splitted = path.split(".")
    return splitted[-2].endswith(base) and splitted[-1].lower() in ["smspec", "fsmspec"]


def _find_file_matching(
    kind: str, case: str, predicate: Callable[[str, str], bool]
) -> str:
    dir, base = os.path.split(case)
    candidates = list(filter(lambda x: predicate(base, x), os.listdir(dir)))
    if not candidates:
        raise ValueError(f"Could not find any {kind} matching case path {case}")
    if len(candidates) > 1:
        raise ValueError(
            f"Ambigous reference to {kind} in {case}, could be any of {candidates}"
        )
    return os.path.join(dir, candidates[0])


def _get_summary_filenames(filepath: str) -> Tuple[str, str]:
    summary = _find_file_matching("unified summary file", filepath, _is_unsmry)
    spec = _find_file_matching("smspec file", filepath, _is_smspec)
    return summary, spec


def read_summary(
    filepath: str, fetch_keys: Sequence[str]
) -> Tuple[List[str], Sequence[datetime], Any]:
    summary, spec = _get_summary_filenames(filepath)
    date_index, start_date, date_units, keys, indices = _read_spec(spec, fetch_keys)
    fetched, time_map = _read_summary(
        summary, start_date, date_units, indices, date_index
    )
    return (keys, time_map, fetched)


def _key2str(key: Union[bytes, str]) -> str:
    ret = key.decode() if isinstance(key, bytes) else key
    assert isinstance(ret, str)
    return ret.strip()


def _read_spec(
    spec: str, fetch_keys: Sequence[str]
) -> Tuple[int, datetime, DateUnit, List[str], npt.NDArray[np.int32]]:
    date = None
    n = None
    nx = None
    ny = None

    arrays: Dict[str, Optional[npt.NDArray[Any]]] = {
        kw: None
        for kw in [
            "WGNAMES ",
            "NUMS    ",
            "KEYWORDS",
            "NUMLX   ",
            "NUMLY   ",
            "NUMLZ   ",
            "LGRNAMES",
            "UNITS   ",
        ]
    }

    if spec.lower().endswith("fsmspec"):
        mode = "rt"
        format = resfo.Format.FORMATTED
    else:
        mode = "rb"
        format = resfo.Format.UNFORMATTED

    with open(spec, mode) as fp:
        for entry in resfo.lazy_read(fp, format):
            if all(
                p is not None
                for p in (
                    [
                        date,
                        n,
                        nx,
                        ny,
                    ]
                    + list(arrays.values())
                )
            ):
                break
            kw = entry.read_keyword()
            if kw in arrays:
                vals = entry.read_array()
                if vals is resfo.MESS or isinstance(vals, resfo.MESS):
                    raise ValueError(f"{kw} in {spec} was MESS")
                arrays[kw] = vals
            if kw == "DIMENS  ":
                vals = entry.read_array()
                if vals is resfo.MESS or isinstance(vals, resfo.MESS):
                    raise ValueError(f"DIMENS in {spec} was MESS")
                size = len(vals)
                n = vals[0] if size > 0 else None
                nx = vals[1] if size > 1 else None
                ny = vals[2] if size > 2 else None
            if kw == "STARTDAT":
                vals = entry.read_array()
                if vals is resfo.MESS or isinstance(vals, resfo.MESS):
                    raise ValueError(f"Startdate in {spec} was MESS")
                size = len(vals)
                day = vals[0] if size > 0 else 0
                month = vals[1] if size > 1 else 0
                year = vals[2] if size > 2 else 0
                hour = vals[3] if size > 3 else 0
                minute = vals[4] if size > 4 else 0
                microsecond = vals[5] if size > 5 else 0
                date = datetime(
                    day=day,
                    month=month,
                    year=year,
                    hour=hour,
                    minute=minute,
                    second=microsecond // 10**6,
                    microsecond=microsecond % 10**6,
                )
    keywords = arrays["KEYWORDS"]
    wgnames = arrays["WGNAMES "]
    nums = arrays["NUMS    "]
    numlx = arrays["NUMLX   "]
    numly = arrays["NUMLY   "]
    numlz = arrays["NUMLZ   "]
    lgr_names = arrays["LGRNAMES"]

    if date is None:
        raise ValueError(f"keyword startdat missing in {spec}")
    if keywords is None:
        raise ValueError(f"keywords missing in {spec}")
    if n is None:
        n = len(keywords)

    indices: List[int] = []
    keys: List[str] = []
    date_index = None

    def optional_get(arr: Optional[npt.NDArray[Any]], idx: int) -> Any:
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
        if key is not None and _should_load_summary_key(key, fetch_keys):
            if key in keys:
                # only keep the index of the last occurrence of a key
                # this is done for backwards compatability
                # and to have unique keys
                indices[keys.index(key)] = i
            else:
                indices.append(i)
                keys.append(key)

    keys_array = np.array(keys)
    rearranged = keys_array.argsort()
    keys_array = keys_array[rearranged]

    indices_array = np.array(indices)[rearranged]

    units = arrays["UNITS   "]
    if units is None:
        raise ValueError(f"keyword units missing in {spec}")
    if date_index is None:
        raise ValueError(f"KEYWORDS did not contain TIME in {spec}")
    if date_index >= len(units):
        raise ValueError(f"Unit missing for TIME in {spec}")

    return (
        date_index,
        date,
        DateUnit[_key2str(units[date_index])],
        list(keys_array),
        indices_array,
    )


def _read_summary(
    summary: str,
    start_date: datetime,
    unit: DateUnit,
    indices: npt.NDArray[np.int32],
    date_index: int,
) -> Tuple[npt.NDArray[np.float32], List[datetime]]:
    if summary.lower().endswith("funsmry"):
        mode = "rt"
        format = resfo.Format.FORMATTED
    else:
        mode = "rb"
        format = resfo.Format.UNFORMATTED

    last_params = None
    values: List[npt.NDArray[np.float32]] = []
    dates: List[datetime] = []

    def read_params() -> None:
        nonlocal last_params, values
        if last_params is not None:
            vals = last_params.read_array()
            if vals is resfo.MESS or isinstance(vals, resfo.MESS):
                raise ValueError(f"PARAMS in {summary} was MESS")
            values.append(vals[indices])
            dates.append(start_date + unit.make_delta(float(vals[date_index])))
            last_params = None

    with open(summary, mode) as fp:
        for entry in resfo.lazy_read(fp, format):
            kw = entry.read_keyword()
            if kw == "PARAMS  ":
                last_params = entry
            if kw == "SEQHDR  ":
                read_params()
        read_params()
    return np.array(values).T, dates


def _should_load_summary_key(data_key: Any, user_set_keys: Sequence[str]) -> bool:
    return any(fnmatch(data_key, key) for key in user_set_keys)
