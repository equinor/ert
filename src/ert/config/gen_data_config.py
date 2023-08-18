from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from sortedcontainers import SortedList

from .response_config import ResponseConfig


@dataclass
class GenDataConfig(ResponseConfig):
    input_file: str = ""
    report_steps: Optional[SortedList] = None

    def __post_init__(self) -> None:
        if isinstance(self.report_steps, list):
            self.report_steps = SortedList(set(self.report_steps))

    def read_from_file(self, run_path: str, _: int) -> xr.Dataset:
        def _read_file(filename: Path, report_step: int) -> xr.Dataset:
            if not filename.exists():
                raise ValueError(f"Missing output file: {filename}")
            data = np.loadtxt(_run_path / filename, ndmin=1)
            active_information_file = _run_path / (str(filename) + "_active")
            if active_information_file.exists():
                index_list = (np.loadtxt(active_information_file) == 0).nonzero()
                data[index_list] = np.nan
            return xr.Dataset(
                {"values": (["report_step", "index"], [data])},
                coords={
                    "index": np.arange(len(data)),
                    "report_step": [report_step],
                },
            )

        errors = []
        datasets = []
        filename_fmt = self.input_file
        _run_path = Path(run_path)
        if self.report_steps is None:
            try:
                datasets.append(_read_file(_run_path / filename_fmt, 0))
            except ValueError as err:
                errors.append(str(err))
        else:
            for report_step in self.report_steps:
                filename = filename_fmt % report_step
                try:
                    datasets.append(_read_file(_run_path / filename, report_step))
                except ValueError as err:
                    errors.append(str(err))
        if errors:
            raise ValueError(f"Error reading GEN_DATA: {self.name}, errors: {errors}")
        return xr.combine_by_coords(datasets)  # type: ignore
