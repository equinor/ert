import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl

from ert.substitutions import substitute_runpath_name

from .parsing import ConfigDict, ConfigValidationError
from .response_config import InvalidResponseFile, ResponseConfig
from .responses_index import responses_index


@dataclass
class EverestObjectivesConfig(ResponseConfig):
    name: str = "everest_objectives"
    has_finalized_keys: bool = True

    @property
    def expected_input_files(self) -> list[str]:
        return self.input_files

    @classmethod
    def from_config_dict(cls, config_dict: ConfigDict) -> Self:
        data = config_dict.get("EVEREST_GEN_DATA", [])
        assert isinstance(data, Iterable)

        keys = []
        input_files = []

        for gen_data in data:
            assert isinstance(gen_data, dict)
            if gen_data.get("type") != "objective":
                continue
            name = gen_data["name"]
            input_file = gen_data.get("input_file")

            if input_file is None:
                raise ConfigValidationError.with_context(
                    f"Missing input file for Everest objective {name!r}",
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

        return cls(
            name="everest_objectives",
            keys=keys,
            input_files=input_files,
        )

    def read_from_file(self, run_path: str, iens: int, iter_: int) -> pl.DataFrame:
        def _read_file(filename: Path) -> pl.DataFrame:
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
                    "index": pl.Series(np.arange(len(data)), dtype=pl.UInt16),
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
        return "everest_objectives"

    @property
    def primary_key(self) -> list[str]:
        return []


responses_index.add_response_type(EverestObjectivesConfig)
