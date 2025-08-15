import os
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
import polars as pl

from ert.substitutions import substitute_runpath_name

from .parsing import ConfigDict, ConfigKeys, ConfigValidationError
from .response_config import InvalidResponseFile, ResponseConfig, ResponseMetadata
from .responses_index import responses_index


class EverestObjectivesConfig(ResponseConfig):
    type: Literal["everest_objectives"] = "everest_objectives"
    name: str = "everest_objectives"
    has_finalized_keys: bool = True

    @property
    def metadata(self) -> list[ResponseMetadata]:
        return [
            ResponseMetadata(
                response_type=self.name,
                response_key=response_key,
                finalized=self.has_finalized_keys,
                filter_on=None,
            )
            for response_key in self.keys
        ]

    @property
    def expected_input_files(self) -> list[str]:
        return self.input_files

    @classmethod
    def from_config_dict(cls, config_dict: ConfigDict) -> Self:
        keys = []
        input_files = []

        for objective in cast(
            list[dict[str, str]], config_dict.get(ConfigKeys.EVEREST_OBJECTIVES, [])
        ):
            name = objective["name"]
            input_file = objective.get("input_file")

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
        return "everest_objectives"

    @property
    def primary_key(self) -> list[str]:
        return []


responses_index.add_response_type(EverestObjectivesConfig)
