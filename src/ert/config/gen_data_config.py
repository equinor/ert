import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
from typing_extensions import Self

from ert.validation import rangestring_to_list

from ._option_dict import option_dict
from .parsing import ConfigDict, ConfigValidationError, ErrorInfo
from .response_config import ResponseConfig
from .responses_index import responses_index


@dataclass
class GenDataConfig(ResponseConfig):
    name: str = "gen_data"
    report_steps_list: List[Optional[List[int]]] = dataclasses.field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if len(self.report_steps_list) == 0:
            self.report_steps_list = [[0] for _ in self.keys]
        else:
            for report_steps in self.report_steps_list:
                if report_steps is not None:
                    report_steps.sort()

    @property
    def expected_input_files(self) -> List[str]:
        expected_files = []
        for input_file, report_steps in zip(self.input_files, self.report_steps_list):
            if report_steps is None:
                expected_files.append(input_file)
            else:
                for report_step in report_steps:
                    expected_files.append(input_file.replace("%d", str(report_step)))

        return expected_files

    @classmethod
    def from_config_dict(cls, config_dict: ConfigDict) -> Optional[Self]:
        gen_data_list = config_dict.get("GEN_DATA", [])  # type: ignore

        keys = []
        input_files = []
        report_steps = []

        for gen_data in gen_data_list:
            options = option_dict(gen_data, 1)
            name = gen_data[0]
            res_file = options.get("RESULT_FILE")

            if res_file is None:
                raise ConfigValidationError.with_context(
                    f"Missing or unsupported RESULT_FILE for GEN_DATA key {name!r}",
                    name,
                )

            _report_steps: Optional[List[int]] = rangestring_to_list(
                options.get("REPORT_STEPS", "")
            )
            _report_steps = sorted(_report_steps) if _report_steps else None
            if os.path.isabs(res_file):
                result_file_context = next(
                    x for x in gen_data if x.startswith("RESULT_FILE:")
                )
                raise ConfigValidationError.with_context(
                    f"The RESULT_FILE:{res_file} setting for {name} is "
                    f"invalid - must be a relative path",
                    result_file_context,
                )

            if _report_steps is None and "%d" in res_file:
                raise ConfigValidationError.from_info(
                    ErrorInfo(
                        message="RESULT_FILES using %d must have REPORT_STEPS:xxxx"
                        " defined. Several report steps separated with ',' "
                        "and ranges with '-' can be listed",
                    ).set_context_keyword(gen_data)
                )

            if _report_steps is not None and "%d" not in res_file:
                result_file_context = next(
                    x for x in gen_data if x.startswith("RESULT_FILE:")
                )
                raise ConfigValidationError.from_info(
                    ErrorInfo(
                        message=f"When configuring REPORT_STEPS:{_report_steps} "
                        "RESULT_FILES must be configured using %d"
                    ).set_context_keyword(result_file_context)
                )

            keys.append(name)
            report_steps.append(_report_steps)
            input_files.append(res_file)

        return cls(
            name="gen_data",
            keys=keys,
            input_files=input_files,
            report_steps_list=report_steps,
        )

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

        _run_path = Path(run_path)
        datasets_per_name = []

        for name, input_file, report_steps in zip(
            self.keys, self.input_files, self.report_steps_list
        ):
            datasets_per_report_step = []
            if report_steps is None:
                try:
                    datasets_per_report_step.append(
                        _read_file(_run_path / input_file, 0)
                    )
                except ValueError as err:
                    errors.append(str(err))
            else:
                for report_step in report_steps:
                    filename = input_file % report_step
                    try:
                        datasets_per_report_step.append(
                            _read_file(_run_path / filename, report_step)
                        )
                    except ValueError as err:
                        errors.append(str(err))

            ds_all_report_steps = xr.concat(
                datasets_per_report_step, dim="report_step"
            ).expand_dims(name=[name])
            datasets_per_name.append(ds_all_report_steps)

        if errors:
            raise ValueError(f"Error reading GEN_DATA: {self.name}, errors: {errors}")

        combined = xr.concat(datasets_per_name, dim="name")
        combined.attrs["response"] = "gen_data"
        return combined

    def get_args_for_key(self, key: str) -> Tuple[Optional[str], Optional[List[int]]]:
        for i, _key in enumerate(self.keys):
            if key == _key:
                return self.input_files[i], self.report_steps_list[i]

        return None, None

    @property
    def response_type(self) -> str:
        return "gen_data"


responses_index.add_response_type(GenDataConfig)
