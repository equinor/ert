from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import polars as pl

from ert.config._reservoir_data_utils import SeismicData

from .parsing import (
    ConfigDict,
    ConfigKeys,
)
from .response_config import InvalidResponseFile, SimulationResponseConfig

logger = logging.getLogger(__name__)


class SeismicConfig(SimulationResponseConfig):
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

    @staticmethod
    def validate_distance_between_responses(df: pl.DataFrame) -> None:
        easts = df["east"].to_numpy()
        norths = df["north"].to_numpy()
        coordinates = np.column_stack([easts, norths])
        too_close_pairs = SeismicData.get_too_close_coordinate_pairs(coordinates)
        if too_close_pairs:
            too_close_coords = [
                (
                    (float(easts[i]), float(norths[i])),
                    (float(easts[j]), float(norths[j])),
                )
                for i, j in too_close_pairs
            ]
            raise InvalidResponseFile(
                "Seismic response coordinates with approximate locations "
                f"{too_close_coords} fall inside of a tolerance radius. All seismic "
                "response coordinates are expected to be more than "
                f"{SeismicData.TOLERANCE * 2} m apart."
            )

    def _collect_response_filepaths(self, run_path: str) -> list[Path]:
        filepaths = []
        for file in self.expected_input_files:
            filepaths.extend(
                SeismicData.resolve_pattern_filepaths(
                    run_path, file, on_error=InvalidResponseFile
                )
            )
        return list(dict.fromkeys(filepaths))

    def read_from_file(self, run_path: str, iens: int, iter_: int) -> pl.DataFrame:
        responses = pl.DataFrame(schema=self.response_schema())
        filepaths = self._collect_response_filepaths(run_path)
        keys = [f.stem for f in filepaths]
        for key, filepath in zip(keys, filepaths, strict=True):
            suffix = filepath.suffix.lower()
            if suffix == ".parquet":
                data = pl.read_parquet(filepath)
            elif suffix == ".csv":
                data = pl.read_csv(filepath)
            else:
                raise InvalidResponseFile(
                    f"Unsupported seismic response file extension {filepath.suffix!r} "
                    f"for {filepath}. Expected '.csv' or '.parquet'."
                )
            df = pl.DataFrame(
                {
                    "response_key": key,
                    "east": data["X_UTME"].cast(pl.Float32),
                    "north": data["Y_UTMN"].cast(pl.Float32),
                    # even though this is a simulated response file, fmu-sim2seis named
                    # the column "OBS"
                    "values": data["OBS"].cast(pl.Float32),
                }
            )
            self.validate_distance_between_responses(df)
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
            has_finalized_keys=False,
        )
