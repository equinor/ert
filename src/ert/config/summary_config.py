from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Set

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
    refcase: Optional[Set[datetime]] = None

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

        keys = []
        time_map = [t.datetime() for t in summary.alloc_time_vector(True)]
        if self.refcase:
            missing = self.refcase.difference(time_map)
            if missing:
                first, last = min(missing), max(missing)
                logger.warning(
                    f"Realization: {iens}, load warning: {len(missing)} "
                    f"inconsistencies in time map, first: Time mismatch for response "
                    f"time: {first}, last: Time mismatch for response time: "
                    f"{last} from: {run_path}/{filename}.UNSMRY"
                )

        user_summary_keys = set(self.keys)
        for key in summary:
            if not self._should_load_summary_key(key, user_summary_keys):
                continue
            keys.append(key)

        data = summary.pandas_frame(column_keys=keys, time_index=time_map).values

        return xr.Dataset(
            {"values": (["name", "time"], data.T)},
            coords={"time": time_map, "name": keys},
        )

    def _should_load_summary_key(self, data_key: Any, user_set_keys: set[str]) -> bool:
        return any(fnmatch(data_key, key) for key in user_set_keys)
