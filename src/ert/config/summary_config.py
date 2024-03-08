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
        self.keys = sorted(set(self.keys))
        if len(self.keys) < 1:
            raise ValueError("SummaryConfig must be given at least one key")

    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset:
        filename = self.input_file.replace("<IENS>", str(iens))
        _, keys, time_map, data = read_summary(f"{run_path}/{filename}", self.keys)
        if len(data) == 0 or len(keys) == 0:
            # https://github.com/equinor/ert/issues/6974
            # There is a bug with storing empty responses so we have
            # to raise an error in that case
            raise ValueError(
                f"Did not find any summary values matching {self.keys} in {filename}"
            )
        ds = xr.Dataset(
            {"values": (["name", "time"], data)},
            coords={"time": time_map, "name": keys},
        )
        return ds.drop_duplicates("time")
