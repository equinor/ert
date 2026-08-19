import fnmatch
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
import scipy as sp


class SeismicData:
    TOLERANCE: ClassVar[float] = 0.1

    @classmethod
    def get_too_close_coordinate_pairs(
        cls, coordinates: Sequence[tuple[float, float]] | np.ndarray
    ) -> set[tuple[int, int]]:
        """Get pairs of indices of coordinates that are within double tolerance of each
        other.

        Double tolerance is used to assure that no response and observation are within
        singular tolerance of each other.
        """

        tree = sp.spatial.KDTree(coordinates)
        return tree.query_pairs(r=cls.TOLERANCE * 2)

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
        extra_cols = []
        if "realization" in responses.columns:
            extra_cols.append("realization")

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

        cols = ["response_key_res", "east_res", "north_res", "east_obs", "north_obs"]
        matched_on_location = (
            candidates.filter(
                (pl.col("east_obs") - pl.col("east_res")).pow(2)
                + (pl.col("north_obs") - pl.col("north_res")).pow(2)
                <= cls.TOLERANCE**2
            )
        ).select(cols + extra_cols)

        left_on = ["response_key", "east", "north", *extra_cols]
        right_on = ["response_key_res", "east_res", "north_res", *extra_cols]
        return (
            responses.join(
                matched_on_location,
                left_on=left_on,
                right_on=right_on,
                how="inner",
            )
            .with_columns(
                east=pl.col("east_obs"),
                north=pl.col("north_obs"),
            )
            .drop(["east_obs", "north_obs"])
        )

    @staticmethod
    def resolve_pattern_filepaths(
        basedir: str,
        pattern: str,
        on_error: Callable[[str], Exception],
    ) -> list[Path]:
        """Resolve all file paths matching a pattern against filenames only.

        The pattern can include a directory path (e.g., "../subdir/obs-*.csv"),
        but the pattern itself is only matched against the filename component.
        Literal glob metacharacters (* ?) should be wrapped in brackets: "test[*].csv".

        Args:
            basedir: Base directory.
            pattern: Pattern with optional directory path and possible glob-style
            wildcards in the filename.
            on_error: Callable to raise a custom exception.

        """
        pattern_path = Path(pattern)
        search_dir = Path(basedir) / pattern_path.parent
        if not search_dir.exists():
            raise on_error(
                f"Directory '{search_dir.absolute()}' does not exist "
                "or is not accessible."
            )

        filename_pattern = pattern_path.name
        has_glob = any(char in filename_pattern for char in "*?[")
        if not has_glob:
            path = search_dir / filename_pattern
            if not path.exists():
                raise on_error(
                    f"File '{path.absolute()}' does not exist or is not accessible."
                )
            return [path]

        compiled_pattern = re.compile(fnmatch.translate(filename_pattern))
        matching_paths = [
            path
            for path in search_dir.iterdir()
            if path.is_file() and compiled_pattern.fullmatch(path.name)
        ]

        if not matching_paths:
            raise on_error(
                f"No files matching pattern '{filename_pattern}' found "
                f"in '{search_dir}'"
            )
        return sorted(matching_paths)
