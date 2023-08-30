from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatch
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from ecl.summary import EclSum

from .response_config import ResponseConfig

if TYPE_CHECKING:
    from typing import Any, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig(ResponseConfig):
    input_file: str
    keys: List[str]
    refcase: Optional[List[datetime]] = None

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
        c_time = summary.alloc_time_vector(True)
        time_map = [t.datetime() for t in c_time]
        if self.refcase:
            missing = []
            for reference_time in self.refcase:
                if reference_time not in time_map:
                    missing.append(reference_time)
            if missing:
                logger.warning(
                    f"Realization: {iens}, load warning: {len(missing)} "
                    f"inconsistencies in time map, first: Time mismatch for response "
                    f"time: {missing[0]}, last: Time mismatch for response time: "
                    f"{missing[-1]} from: {run_path}/{filename}.UNSMRY"
                )

        user_summary_keys = set(self.keys)
        for key in summary:
            if not self._should_load_summary_key(key, user_summary_keys):
                continue
            keys.append(key)

            np_vector = np.zeros(len(time_map))
            summary._init_numpy_vector_interp(
                key,
                c_time,
                np_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            data.append(np_vector)

        return xr.Dataset(
            {"values": (["name", "time"], data)},
            coords={"time": time_map, "name": keys},
        )

    def _should_load_summary_key(self, data_key: Any, user_set_keys: set[str]) -> bool:
        return any(fnmatch(data_key, key) for key in user_set_keys)
