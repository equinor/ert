from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Set, Union

import xarray as xr

from ._read_summary import read_summary
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
        _, keys, time_map, data = read_summary(f"{run_path}/{filename}", self.keys)
        ds = xr.Dataset(
            {"values": (["name", "time"], data)},
            coords={"time": time_map, "name": keys},
        )
        return ds.drop_duplicates("time")
