from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from ecl.summary import EclSum

from ert.config.response_config import ResponseConfig

if TYPE_CHECKING:
    from typing import Any, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig(ResponseConfig):
    input_file: str
    keys: List[str]
    refcase: Optional[EclSum] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SummaryConfig):
            return False
        refcase_equal = True
        if self.refcase:
            refcase_equal = bool(
                other.refcase and self.refcase.case == other.refcase.case
            )

        return all(
            [
                self.name == other.name,
                self.input_file == other.input_file,
                self.keys == other.keys,
                refcase_equal,
            ]
        )

    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset:
        filename = self.input_file.replace("<IENS>", str(iens))
        try:
            summary = EclSum(
                f"{run_path}/{filename}",
                include_restart=False,
                lazy_load=False,
            )
        except IOError as e:
            raise ValueError(
                "Could not find SUMMARY file or using non unified SUMMARY "
                f"file from: {run_path}/{filename}.UNSMRY",
            ) from e

        data = []
        keys = []
        time_map = summary.alloc_time_vector(True)
        axis = [t.datetime() for t in time_map]
        if self.refcase:
            existing_time_map = self.refcase.alloc_time_vector(True)
            missing = []
            for step, (response_t, reference_t) in enumerate(
                zip(time_map, existing_time_map)
            ):
                if response_t not in existing_time_map:
                    missing.append((response_t, reference_t, step + 1))
            if missing:
                logger.warning(
                    f"Realization: {iens}, load warning: {len(missing)} "
                    "inconsistencies in time map, first: "
                    f"Time mismatch for step: {missing[0][2]}, response time: "
                    f"{missing[0][0]}, reference case: {missing[0][1]}, last: Time "
                    f"mismatch for step: {missing[-1][2]}, response time: "
                    f"{missing[-1][0]}, reference case: {missing[-1][1]} "
                    f"from: {run_path}/{filename}.UNSMRY"
                )

        user_summary_keys = set(self.keys)
        for key in summary:
            if not self._should_load_summary_key(key, user_summary_keys):
                continue
            keys.append(key)

            np_vector = np.zeros(len(time_map))
            summary._init_numpy_vector_interp(
                key,
                time_map,
                np_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            data.append(np_vector)

        return xr.Dataset(
            {"values": (["name", "time"], data)},
            coords={"time": axis, "name": keys},
        )

    def _should_load_summary_key(self, data_key: Any, user_set_keys: set[str]) -> bool:
        return any(fnmatch(data_key, key) for key in user_set_keys)
