import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Self, cast

import numpy as np
import polars as pl
from pydantic import Field

from ert.substitutions import substitute_runpath_name
from ert.validation import rangestring_to_list

from .parsing import ConfigDict, ConfigValidationError, ErrorInfo
from .response_config import (
    InvalidResponseFile,
    ResponseConfig,
    ResponseMetadata,
)
from .responses_index import responses_index


class GenDataConfig(ResponseConfig):
    type: Literal["gen_data"] = "gen_data"
    name: str = "gen_data"
    report_steps_list: list[list[int] | None] = Field(default_factory=list)
    has_finalized_keys: bool = True

    @property
    def metadata(self) -> list[ResponseMetadata]:
        return [
            ResponseMetadata(
                response_type=self.name,
                response_key=response_key,
                finalized=self.has_finalized_keys,
                filter_on={"report_step": report_steps}
                if report_steps is not None
                else {"report_step": [0]},
            )
            for response_key, report_steps in zip(
                self.keys, self.report_steps_list, strict=False
            )
        ]

    def model_post_init(self, ctx: Any) -> None:
        if len(self.report_steps_list) == 0:
            self.report_steps_list = [[0] for _ in self.keys]
        else:
            for report_steps in self.report_steps_list:
                if report_steps is not None:
                    report_steps.sort()

    @property
    def expected_input_files(self) -> list[str]:
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
    def from_config_dict(cls, config_dict: ConfigDict) -> Self:
        gen_data_list = config_dict.get("GEN_DATA", [])
        assert isinstance(gen_data_list, Iterable)

        keys = []
        input_files = []
        report_steps = []

        for gen_data in gen_data_list:
            options = cast(dict[str, str], gen_data[1])
            name = gen_data[0]
            res_file = options.get("RESULT_FILE")
            report_steps_value = options.get("REPORT_STEPS", "")

            if res_file is None:
                raise ConfigValidationError.with_context(
                    f"Missing RESULT_FILE for GEN_DATA key {name!r}",
                    name,
                )
            try:
                report_steps_: list[int] | None = rangestring_to_list(
                    report_steps_value
                )
            except ValueError as e:
                raise ConfigValidationError.with_context(
                    f"The REPORT_STEPS setting: {report_steps_value} for {name} is "
                    'invalid - must be a valid range string: e.g.: "0-1, 4-6, 8"',
                    gen_data,
                ) from e

            report_steps_ = sorted(report_steps_) if report_steps_ else None
            if os.path.isabs(res_file):
                raise ConfigValidationError.with_context(
                    f"The RESULT_FILE:{res_file} setting for {name} is "
                    f"invalid - must be a relative path",
                    name,
                )

            if report_steps_ is None and "%d" in res_file:
                raise ConfigValidationError.from_info(
                    ErrorInfo(
                        message="RESULT_FILES using %d must have REPORT_STEPS:xxxx"
                        " defined. Several report steps separated with ',' "
                        "and ranges with '-' can be listed",
                    ).set_context_keyword(gen_data)
                )

            if report_steps_ is not None and "%d" not in res_file:
                raise ConfigValidationError.from_info(
                    ErrorInfo(
                        message=f"When configuring REPORT_STEPS:{report_steps_} "
                        "RESULT_FILES must be configured using %d"
                    ).set_context_keyword(name)
                )

            keys.append(name)
            report_steps.append(report_steps_)
            input_files.append(res_file)

        return cls(
            name="gen_data",
            keys=keys,
            input_files=input_files,
            report_steps_list=report_steps,
        )

    def read_from_file(self, run_path: str, iens: int, iter_: int) -> pl.DataFrame:
        def _read_file(filename: Path, report_step: int) -> pl.DataFrame:
            try:
                data = np.loadtxt(filename, ndmin=1)
            except ValueError as err:
                raise InvalidResponseFile(str(err)) from err
            active_information_file = filename.parent / (filename.name + "_active")
            if active_information_file.exists():
                try:
                    active_list = np.loadtxt(active_information_file)
                except ValueError as err:
                    raise InvalidResponseFile(str(err)) from err
                data[active_list == 0] = np.nan
            return pl.DataFrame(
                {
                    "report_step": pl.Series(
                        np.full(len(data), report_step), dtype=pl.UInt16
                    ),
                    "index": pl.Series(np.arange(len(data)), dtype=pl.UInt16),
                    "values": pl.Series(data, dtype=pl.Float32),
                }
            )

        errors = []

        run_path_ = Path(run_path)
        datasets_per_name = []

        for name, input_file, report_steps in zip(
            self.keys, self.input_files, self.report_steps_list, strict=False
        ):
            datasets_per_report_step = []
            if report_steps is None:
                try:
                    filename = substitute_runpath_name(input_file, iens, iter_)
                    datasets_per_report_step.append(_read_file(run_path_ / filename, 0))
                except (InvalidResponseFile, FileNotFoundError) as err:
                    errors.append(err)
            else:
                for report_step in report_steps:
                    filename = substitute_runpath_name(
                        input_file % report_step, iens, iter_
                    )
                    try:
                        datasets_per_report_step.append(
                            _read_file(run_path_ / filename, report_step)
                        )
                    except (InvalidResponseFile, FileNotFoundError) as err:
                        errors.append(err)

            if len(datasets_per_report_step) > 0:
                ds_all_report_steps = pl.concat(datasets_per_report_step)
                ds_all_report_steps.insert_column(
                    0, pl.Series("response_key", [name] * len(ds_all_report_steps))
                )
                datasets_per_name.append(ds_all_report_steps)

        if errors:
            if all(isinstance(err, FileNotFoundError) for err in errors):
                raise FileNotFoundError(
                    "Could not find one or more files/directories while reading "
                    f"GEN_DATA {self.name}: {','.join([str(err) for err in errors])}"
                )
            else:
                raise InvalidResponseFile(
                    "Error reading GEN_DATA "
                    f"{self.name}, errors: {','.join([str(err) for err in errors])}"
                )

        combined = pl.concat(datasets_per_name)
        return combined

    def get_args_for_key(self, key: str) -> tuple[str | None, list[int] | None]:
        for i, _key in enumerate(self.keys):
            if key == _key:
                return self.input_files[i], self.report_steps_list[i]

        return None, None

    @property
    def response_type(self) -> str:
        return "gen_data"

    @property
    def primary_key(self) -> list[str]:
        return ["report_step", "index"]


responses_index.add_response_type(GenDataConfig)
