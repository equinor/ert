from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Self

import polars as pl

from ert.substitutions import substitute_runpath_name

from .parsing import (
    ConfigDict,
    ConfigKeys,
)
from .response_config import InvalidResponseFile, ResponseConfig
from .responses_index import responses_index

logger = logging.getLogger(__name__)


class SeismicConfig(ResponseConfig):
    """Configuration for responses from https://github.com/equinor/fmu-sim2seis

    Reads files created by MAP_ATTRIBUTES forward model. Files should have columns
    X_UTME, Y_UTMN and OBS (value). The response key is derived from the filename, and
    the match keys are east and north coordinates.
    """

    name: str = "seismic"
    type: Literal["seismic"] = "seismic"

    @property
    def expected_input_files(self) -> list[str]:
        return self.input_files

    @staticmethod
    def response_schema() -> dict[str, Any]:
        return {
            "response_key": pl.String,
            "east": pl.Float32,
            "north": pl.Float32,
            "values": pl.Float32,
        }

    def read_from_file(self, run_path: str, iens: int, iter_: int) -> pl.DataFrame:
        responses = pl.DataFrame(schema=self.response_schema())
        for key, file in zip(self.keys, self.expected_input_files, strict=True):
            filepath_runpath_relative = substitute_runpath_name(file, iens, iter_)
            filepath = Path(run_path) / filepath_runpath_relative
            if not filepath.exists():
                raise InvalidResponseFile(
                    f"Expected seismic response file {filepath} does not exist."
                )
            csv = pl.read_csv(filepath)
            df = pl.DataFrame(
                {
                    "response_key": key,
                    "east": csv["X_UTME"].cast(pl.Float32),
                    "north": csv["Y_UTMN"].cast(pl.Float32),
                    # even though this is a simulated response file, fmu-sim2seis named
                    # the column "OBS"
                    "values": csv["OBS"].cast(pl.Float32),
                }
            )
            duplicates = (
                df.group_by(["east", "north"])
                .agg(pl.len().alias("count"))
                .filter(pl.col("count") > 1)
            )
            duplicates_str = "\n".join(
                f"  east={d['east']}, north={d['north']}, count={d['count']}"
                for d in duplicates.to_dicts()
            )
            if len(duplicates) > 0:
                raise InvalidResponseFile(
                    "Seismic response coordinates were not unique (after rounding "
                    "from f64 to f32). Approximate locations are:\n"
                    f"{duplicates_str}"
                )

            responses = pl.concat([responses, df], how="vertical")
        return self._assert_schema(responses, self.response_schema())

    @property
    def response_type(self) -> str:
        return "seismic"

    @property
    def match_key(self) -> list[str]:
        return ["east", "north"]

    @classmethod
    def from_config_dict(cls, config_dict: ConfigDict) -> Self | None:
        files: list[str] = config_dict.get(ConfigKeys.SEISMIC, [])
        for file in files:
            assert isinstance(file, str), f"Expected str, got {type(file)}: {file!r}"
        return cls(
            name="seismic",
            input_files=files,
            keys=[Path(f).stem for f in files],
        )


responses_index.add_response_type(SeismicConfig)
