import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr
from typing_extensions import Self

from ert.validation import rangestring_to_list

from ._option_dict import option_dict
from .parsing import ConfigValidationError, ErrorInfo
from .response_config import ResponseConfig


@dataclass
class GenDataConfig(ResponseConfig):
    input_file: str = ""
    report_steps: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if isinstance(self.report_steps, list):
            self.report_steps = list(set(self.report_steps))

    @classmethod
    def from_config_list(cls, gen_data: List[str]) -> Self:
        options = option_dict(gen_data, 1)
        name = gen_data[0]
        res_file = options.get("RESULT_FILE")

        if res_file is None:
            raise ConfigValidationError.with_context(
                f"Missing or unsupported RESULT_FILE for GEN_DATA key {name!r}", name
            )

        report_steps: Optional[List[int]] = rangestring_to_list(
            options.get("REPORT_STEPS", "")
        )
        report_steps = sorted(report_steps) if report_steps else None
        if os.path.isabs(res_file):
            result_file_context = next(
                x for x in gen_data if x.startswith("RESULT_FILE:")
            )
            raise ConfigValidationError.with_context(
                f"The RESULT_FILE:{res_file} setting for {name} is "
                f"invalid - must be a relative path",
                result_file_context,
            )

        if report_steps is None and "%d" in res_file:
            raise ConfigValidationError.from_info(
                ErrorInfo(
                    message="RESULT_FILES using %d must have REPORT_STEPS:xxxx"
                    " defined. Several report steps separated with ',' "
                    "and ranges with '-' can be listed",
                ).set_context_keyword(gen_data)
            )

        if report_steps is not None and "%d" not in res_file:
            result_file_context = next(
                x for x in gen_data if x.startswith("RESULT_FILE:")
            )
            raise ConfigValidationError.from_info(
                ErrorInfo(
                    message=f"When configuring REPORT_STEPS:{report_steps} "
                    "RESULT_FILES must be configured using %d"
                ).set_context_keyword(result_file_context)
            )
        return cls(name=name, input_file=res_file, report_steps=report_steps)

    def read_from_file(self, run_path: str, _: int) -> xr.Dataset:
        def _read_file(filename: Path, report_step: int) -> xr.Dataset:
            if not filename.exists():
                raise ValueError(f"Missing output file: {filename}")
            data = np.loadtxt(_run_path / filename, ndmin=1)
            active_information_file = _run_path / (str(filename) + "_active")
            if active_information_file.exists():
                active_list = np.loadtxt(active_information_file)
                data[active_list == 0] = np.nan
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
                filename = filename_fmt % report_step  # noqa
                try:
                    datasets.append(_read_file(_run_path / filename, report_step))
                except ValueError as err:
                    errors.append(str(err))
        if errors:
            raise ValueError(f"Error reading GEN_DATA: {self.name}, errors: {errors}")
        return xr.combine_nested(datasets, concat_dim="report_step")
