import os
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
import polars as pl

from ert.config.parsing import ConfigKeys
from ert.substitutions import substitute_runpath_name

from .parsing import ConfigDict, ConfigValidationError
from .response_config import InvalidResponseFile, ResponseConfig, ResponseMetadata
from .responses_index import responses_index


class EverestConstraintsConfig(ResponseConfig):
    @property
    def metadata(self) -> list[ResponseMetadata]:
        return [
            ResponseMetadata(
                response_type=self.name,
                response_key=response_key,
                filter_on=None,
            )
            for response_key in self.keys
        ]

    type: Literal["everest_constraints"] = "everest_constraints"
    name: str = "everest_constraints"
    has_finalized_keys: bool = True
    auto_scales: list[bool]
    scales: list[float | None]
    upper_bounds: list[float | None]
    lower_bounds: list[float | None]

    @property
    def expected_input_files(self) -> list[str]:
        return self.input_files

    @classmethod
    def from_config_dict(cls, config_dict: ConfigDict) -> Self:
        keys = []
        input_files = []
        auto_scales = []
        scales = []
        upper_bounds = []
        lower_bounds = []

        for constraint in cast(
            list[dict[str, str]], config_dict.get(ConfigKeys.EVEREST_CONSTRAINTS, [])
        ):
            name = constraint["name"]
            input_file = constraint.get("input_file")
            auto_scale = constraint.get("auto_scale")
            scale = constraint.get("scale")
            upper_bound = constraint.get("upper_bound")
            lower_bound = constraint.get("lower_bound")

            if input_file is None:
                raise ConfigValidationError.with_context(
                    f"Missing input file for Everest constraint {name!r}",
                    name,
                )
            if os.path.isabs(input_file):
                raise ConfigValidationError.with_context(
                    f"The input file:{input_file} for {name} is "
                    f"invalid - must be a relative path",
                    name,
                )

            keys.append(name)
            input_files.append(input_file)
            auto_scales.append(auto_scale)
            scales.append(scale)
            upper_bounds.append(upper_bound)
            lower_bounds.append(lower_bound)

        return cls(
            name="everest_constraints",
            keys=keys,
            input_files=input_files,
            auto_scales=auto_scales,
            scales=scales,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
        )

    def read_from_file(self, run_path: str, iens: int, iter_: int) -> pl.DataFrame:
        def _read_file(filename: Path) -> pl.DataFrame:
            try:
                data = np.loadtxt(filename, ndmin=1)
            except ValueError as err:
                raise InvalidResponseFile(str(err)) from err
            return pl.DataFrame(
                {
                    "values": pl.Series(data, dtype=pl.Float32),
                }
            )

        errors = []

        run_path_ = Path(run_path)
        datasets_per_name = []

        for name, input_file in zip(self.keys, self.input_files, strict=False):
            datasets = []
            try:
                filename = substitute_runpath_name(input_file, iens, iter_)
                datasets.append(_read_file(run_path_ / filename))
            except (InvalidResponseFile, FileNotFoundError) as err:
                errors.append(err)

            if len(datasets) > 0:
                combined_ds = pl.concat(datasets)
                combined_ds.insert_column(
                    0, pl.Series("response_key", [name] * len(combined_ds))
                )
                datasets_per_name.append(combined_ds)

        if errors:
            if all(isinstance(err, FileNotFoundError) for err in errors):
                raise FileNotFoundError(
                    "Could not find one or more files/directories while reading "
                    f"{self.name}: {','.join([str(err) for err in errors])}"
                )
            else:
                raise InvalidResponseFile(
                    "Error reading "
                    f"{self.name}, errors: {','.join([str(err) for err in errors])}"
                )

        combined = pl.concat(datasets_per_name)
        return combined

    @property
    def response_type(self) -> str:
        return "everest_constraints"

    @property
    def primary_key(self) -> list[str]:
        return []


responses_index.add_response_type(EverestConstraintsConfig)
