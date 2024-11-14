import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import polars
from typing_extensions import Self

from ert.validation import rangestring_to_list

from ._option_dict import option_dict
from .parsing import ConfigDict, ConfigValidationError, ErrorInfo
from .response_config import InvalidResponseFile, ResponseConfig
from .responses_index import responses_index


@dataclass
class GenDataConfig(ResponseConfig):
    name: str = "gen_data"
    report_steps_list: List[Optional[List[int]]] = dataclasses.field(
        default_factory=list
    )
    has_finalized_keys: bool = True

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
        for input_file, report_steps in zip(
            self.input_files, self.report_steps_list, strict=False
        ):
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
            report_steps_value = options.get("REPORT_STEPS", "")

            if res_file is None:
                raise ConfigValidationError.with_context(
                    f"Missing or unsupported RESULT_FILE for GEN_DATA key {name!r}",
                    name,
                )
            try:
                _report_steps: Optional[List[int]] = rangestring_to_list(
                    report_steps_value
                )
            except ValueError as e:
                raise ConfigValidationError.with_context(
                    f"The REPORT_STEPS setting: {report_steps_value} for {name} is invalid"
                    ' - must be a valid range string: e.g.: "0-1, 4-6, 8"',
                    gen_data,
                ) from e

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

    def read_from_file(self, run_path: str, _: int) -> polars.DataFrame:
        def _read_file(filename: Path, report_step: int) -> polars.DataFrame:
            try:
                data = np.loadtxt(_run_path / filename, ndmin=1)
            except ValueError as err:
                raise InvalidResponseFile(str(err)) from err
            active_information_file = _run_path / (str(filename) + "_active")
            if active_information_file.exists():
                try:
                    active_list = np.loadtxt(active_information_file)
                except ValueError as err:
                    raise InvalidResponseFile(str(err)) from err
                data[active_list == 0] = np.nan
            return polars.DataFrame(
                {
                    "report_step": polars.Series(
                        np.full(len(data), report_step), dtype=polars.UInt16
                    ),
                    "index": polars.Series(np.arange(len(data)), dtype=polars.UInt16),
                    "values": polars.Series(data, dtype=polars.Float32),
                }
            )

        errors = []

        _run_path = Path(run_path)
        datasets_per_name = []

        for name, input_file, report_steps in zip(
            self.keys, self.input_files, self.report_steps_list, strict=False
        ):
            datasets_per_report_step = []
            if report_steps is None:
                try:
                    datasets_per_report_step.append(
                        _read_file(_run_path / input_file, 0)
                    )
                except (InvalidResponseFile, FileNotFoundError) as err:
                    errors.append(err)
            else:
                for report_step in report_steps:
                    filename = input_file % report_step
                    try:
                        datasets_per_report_step.append(
                            _read_file(_run_path / filename, report_step)
                        )
                    except (InvalidResponseFile, FileNotFoundError) as err:
                        errors.append(err)

            if len(datasets_per_report_step) > 0:
                ds_all_report_steps = polars.concat(datasets_per_report_step)
                ds_all_report_steps.insert_column(
                    0, polars.Series("response_key", [name] * len(ds_all_report_steps))
                )
                datasets_per_name.append(ds_all_report_steps)

        if errors:
            if all(isinstance(err, FileNotFoundError) for err in errors):
                raise FileNotFoundError(
                    "Could not find one or more files/directories while reading GEN_DATA"
                    f" {self.name}: {','.join([str(err) for err in errors])}"
                )
            else:
                raise InvalidResponseFile(
                    "Error reading GEN_DATA "
                    f"{self.name}, errors: {','.join([str(err) for err in errors])}"
                )

        combined = polars.concat(datasets_per_name)
        return combined

    def get_args_for_key(self, key: str) -> Tuple[Optional[str], Optional[List[int]]]:
        for i, _key in enumerate(self.keys):
            if key == _key:
                return self.input_files[i], self.report_steps_list[i]

        return None, None

    @property
    def response_type(self) -> str:
        return "gen_data"

    @property
    def primary_key(self) -> List[str]:
        return ["report_step", "index"]


responses_index.add_response_type(GenDataConfig)
