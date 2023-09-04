from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import xarray as xr
from ecl.summary import EclSum

from ert._clib._read_summary import read_summary  # pylint: disable=import-error

from .response_config import ResponseConfig

if TYPE_CHECKING:
    from typing import List, Optional


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

        c_time = summary.alloc_time_vector(True)
        time_map = [t.datetime() for t in c_time]
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

        summary_data = read_summary(summary, self.keys)
        summary_data.sort(key=lambda x: x[0])
        data = [d for _, d in summary_data]
        keys = [k for k, _ in summary_data]

        ds = xr.Dataset(
            {"values": (["name", "time"], data)},
            coords={"time": time_map, "name": keys},
        )
        return ds.drop_duplicates(["time"])
