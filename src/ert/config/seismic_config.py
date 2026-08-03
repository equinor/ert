from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

import numpy as np
import polars as pl
import scipy as sp

from ert.substitutions import substitute_runpath_name

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

    TOLERANCE: ClassVar[float] = 0.1

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
        tree = sp.spatial.KDTree(coordinates)
        too_close_pairs = tree.query_pairs(r=SeismicConfig.TOLERANCE * 2)
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
                f"{SeismicConfig.TOLERANCE * 2} m apart."
            )

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
        )

    @classmethod
    def use_observation_locations_in_respective_responses(
        cls, responses: pl.DataFrame, observations: pl.DataFrame
    ) -> pl.DataFrame:
        """Unify response and observation locations.

        Replace the east and north coordinates in the response dataframe with the
        corresponding coordinates from the observation dataframe, if they are within the
        tolerance radius. Drop response otherwise. Required to match responses to
        observations regardless of location precision discrepancies.
        """
        candidates = responses.rename(
            {
                "east": "east_res",
                "north": "north_res",
                "response_key": "response_key_res",
            }
        ).join_where(
            observations.rename(
                {
                    "east": "east_obs",
                    "north": "north_obs",
                    "response_key": "response_key_obs",
                }
            ),
            pl.col("response_key_res") == pl.col("response_key_obs"),
            pl.col("east_obs") >= pl.col("east_res") - cls.TOLERANCE,
            pl.col("east_obs") <= pl.col("east_res") + cls.TOLERANCE,
            pl.col("north_obs") >= pl.col("north_res") - cls.TOLERANCE,
            pl.col("north_obs") <= pl.col("north_res") + cls.TOLERANCE,
        )
        matched_on_location = (
            candidates.filter(
                (pl.col("east_obs") - pl.col("east_res")).pow(2)
                + (pl.col("north_obs") - pl.col("north_res")).pow(2)
                <= cls.TOLERANCE**2
            )
        ).select(["response_key_res", "east_res", "north_res", "east_obs", "north_obs"])

        return (
            responses.join(
                matched_on_location,
                left_on=["response_key", "east", "north"],
                right_on=["response_key_res", "east_res", "north_res"],
                how="inner",
            )
            .with_columns(
                east=pl.col("east_obs"),
                north=pl.col("north_obs"),
            )
            .drop(["east_obs", "north_obs"])
        )
