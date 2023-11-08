from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Set, Union

import xarray as xr
from resdata.summary import Summary

from ert._clib._read_summary import (  # pylint: disable=import-error
    read_dates,
    read_summary,
)

from .response_config import ResponseConfig

if TYPE_CHECKING:
    from typing import List


logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig(ResponseConfig):
    input_file: str
    keys: List[str]
    refcase: Union[Set[datetime], List[str], None] = None

    def __post_init__(self) -> None:
        if isinstance(self.refcase, list):
            self.refcase = {datetime.fromisoformat(val) for val in self.refcase}

    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset:
        filename = self.input_file.replace("<IENS>", str(iens))
        try:
            summary = Summary(
                f"{run_path}/{filename}",
                include_restart=False,
                lazy_load=False,
            )
        except IOError as e:
            raise ValueError(
                "Could not find SUMMARY file or using non unified SUMMARY "
                f"file from: {run_path}/{filename}.UNSMRY",
            ) from e
        time_map = read_dates(summary)
        if self.refcase:
            assert isinstance(self.refcase, set)
            missing = self.refcase.difference(time_map)
            if missing:
                first, last = min(missing), max(missing)
                logger.warning(
                    f"Realization: {iens}, load warning: {len(missing)} "
                    f"inconsistencies in time map, first: Time mismatch for response "
                    f"time: {first}, last: Time mismatch for response time: "
                    f"{last} from: {run_path}/{filename}.UNSMRY"
                )

        summary_data = read_summary(summary, list(set(self.keys)))
        summary_data.sort(key=lambda x: x[0])
        data = [d for _, d in summary_data]
        keys = [k for k, _ in summary_data]
        ds = xr.Dataset(
            {"values": (["name", "time"], data)},
            coords={"time": time_map, "name": keys},
        )
        return ds.drop_duplicates("time")
