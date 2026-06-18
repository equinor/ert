from __future__ import annotations

import datetime
import fnmatch
import logging
import os
import re
from collections import defaultdict
from dataclasses import InitVar, dataclass
from functools import lru_cache
from pathlib import Path
from typing import IO, Any, Literal, override

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import Field
from resfo_utilities import (
    CornerpointGrid,
    InvalidEgridFileError,
    InvalidRFTError,
    RFTReader,
)

from ert.substitutions import substitute_runpath_name

from .parsing import (
    ConfigDict,
    ConfigKeys,
    ConfigValidationError,
    ConfigWarning,
    parse_zonemap,
)
from .response_config import (
    InvalidResponseFile,
    ResponseConfig,
    _warn_about_missing_responses,
)
from .responses_index import responses_index

logger = logging.getLogger(__name__)


# A Point in UTM/TVD coordinates
type Point = tuple[float, float, float]
# Index to a cell in a grid
type GridIndex = tuple[int, int, int]
type ZoneName = str
type WellName = str
type DateString = str
type RFTProperty = str


@dataclass(frozen=True)
class WellConnectionCell:
    grid_index: GridIndex | None
    cell_center: Point | None


# The egrid and zonemap are needed in both ``read_from_file`` and
# ``obtain_location_metadata``.
# These functions are called just after each other for the same realization/file path in
# ``_write_responses_to_storage`` (see ``local_ensemble.py``). Caching them avoids
# reading and parsing the same files twice. ``maxsize=1`` since we don't want to keep
# multiple large objects in memory (especially grid objects can be large), and it is
# sufficient as ``_write_responses_to_storage`` is only called for one realization at a
# time.


@lru_cache(maxsize=1)
def _read_egrid(egrid_file: str) -> CornerpointGrid:
    try:
        return CornerpointGrid.read_egrid(egrid_file)
    except (OSError, InvalidEgridFileError) as err:
        raise InvalidResponseFile(f"Could not read grid file: {err}") from err


@lru_cache(maxsize=1)
def _get_zonemap(zonemap_path: Path) -> dict[int, list[str]]:
    return parse_zonemap(str(zonemap_path), zonemap_path.read_text(encoding="utf-8"))


class RFTConfig(ResponseConfig):
    """:term:`RFT` response from a :term:`reservoir simulator`.

    RFTConfig is the configuration of responses in the <RUNPATH>/<ECLBASE>.RFT
    file which may be generated from a reservoir simulator forward model step.

    The file contains values for grid cells along a wellpath (see RFTReader for
    details). RFTConfig will match the values against the given :term:`UTM`/:term:`TVD`
    locations.

    Parameters:
        data_to_read: dictionary of the values that should be read from the rft file.
        zonemap: The mapping from grid layer index to zone name.
        approximate_missing_values: option to fill missing values by interpolation or
        extrapolation from points within same zone.
    """

    type: Literal["rft"] = "rft"
    name: str = "rft"
    has_finalized_keys: bool = False
    data_to_read: dict[WellName, dict[DateString, list[RFTProperty]]] = Field(
        default_factory=dict
    )
    zonemap: Path | None = None
    approximate_missing_values: bool = False

    @property
    def expected_input_files(self) -> list[str]:
        base = self.input_files[0]
        if base.upper().endswith(".DATA"):
            # For backwards compatibility, it is
            # allowed to give REFCASE and ECLBASE both
            # with and without .DATA extensions
            base = base[:-5]

        return [f"{base}.RFT"]

    @staticmethod
    def _rft_filepath(base_name: str, run_path: str, iens: int, iter_: int) -> str:
        base_name = substitute_runpath_name(base_name, iens, iter_)
        if base_name.upper().endswith(".DATA"):
            # For backwards compatibility, it is
            # allowed to give REFCASE and ECLBASE both
            # with and without .DATA extensions
            base_name = base_name[:-5]

        return f"{run_path}/{base_name}"

    @staticmethod
    def _ergrid_filepath(rft_filepath: str) -> str:
        grid_filepath = rft_filepath
        if grid_filepath.upper().endswith(".RFT"):
            grid_filepath = grid_filepath[:-4]
        grid_filepath += ".EGRID"
        return grid_filepath

    @staticmethod
    def _zonemap_filepath(
        base_path: Path, run_path: str, iens: int, iter_: int
    ) -> Path:
        zonemap_filepath = Path(substitute_runpath_name(str(base_path), iens, iter_))
        if not base_path.is_absolute():
            zonemap_filepath = Path(run_path) / zonemap_filepath
        return zonemap_filepath

    @staticmethod
    def _map_locations_to_cells(
        egrid_file: str | os.PathLike[str] | IO[Any], locations: list[Point]
    ) -> dict[Point, WellConnectionCell]:
        """
        For each location, find the corresponding connected grid cell and its center,
        if it exists.
        """

        location_cell_map: dict[Point, WellConnectionCell] = {}
        if not locations:
            return location_cell_map
        grid = _read_egrid(str(egrid_file))

        for location, cell in zip(
            locations,
            grid.find_cell_containing_point(locations),
            strict=True,
        ):
            if cell is None:
                location_cell_map[location] = WellConnectionCell(
                    None,
                    None,
                )
            else:
                location_cell_map[location] = WellConnectionCell(
                    (cell[0] + 1, cell[1] + 1, cell[2] + 1),
                    tuple(grid.cell_corners(*cell).mean(axis=0)),
                )
        return location_cell_map

    @staticmethod
    def _assert_schema(df: pl.DataFrame, schema: dict[str, Any]) -> pl.DataFrame:
        if df.schema != schema:
            msg = f"Expected schema {schema}, got {df.schema}."
            raise AssertionError(msg)
        return df

    @dataclass(frozen=True)
    class ValidRFTEntry:
        property_values: dict[RFTProperty, npt.NDArray[np.float32]]
        # See :term:`well connection` in glossary
        well_connection_cells: npt.NDArray[np.integer]

        filepath: InitVar[str]
        well: InitVar[str]
        date: InitVar[datetime.date]

        def __post_init__(self, filepath: str, well: str, date: datetime.date) -> None:
            num_conns = len(self.well_connection_cells)
            for rft_property, values in self.property_values.items():
                num_values = len(values)
                if num_values != num_conns:
                    raise InvalidResponseFile(
                        "Could not read RFT from "
                        f"{filepath}: "
                        f"RFT property {rft_property} for well {well} "
                        f"at {date.isoformat()} has {num_values} "
                        f"value{'s' if num_values != 1 else ''} "
                        f"but {num_conns} well "
                        f"connection{'s' if num_conns != 1 else ''}"
                    )

    def _scan_rft(
        self, filepath: str
    ) -> dict[tuple[WellName, datetime.date], RFTConfig.ValidRFTEntry]:
        # This is a somewhat complicated optimization in order to
        # support wildcards in well names, dates and properties
        # A python for loop is too slow so we use a compiled regex
        # instead

        sep = "\x31"

        def _translate(pat: str) -> str:
            """Translates fnmatch pattern to match anywhere"""
            return fnmatch.translate(pat).replace("\\z", "").replace("\\Z", "")

        def _props_matcher(props: list[str]) -> str:
            """Regex for matching given props _and_ DEPTH"""
            pattern = f"({'|'.join(_translate(p) for p in props)})"
            if re.fullmatch(pattern, "DEPTH") is None:
                return f"({'|'.join(_translate(p) for p in [*props, 'DEPTH'])})"
            return pattern

        matcher = re.compile(
            "|".join(
                "("
                + re.escape(sep).join(
                    (
                        _translate(well),
                        _translate(time),
                        _props_matcher(props),
                    )
                )
                + ")"
                for well, inner_dict in self.data_to_read.items()
                for time, props in inner_dict.items()
            )
        )

        rft_data: dict[tuple[WellName, datetime.date], RFTConfig.ValidRFTEntry] = {}
        try:  # noqa: PLW0717
            with RFTReader.open(filepath) as rft:
                for entry in rft:
                    date = entry.date
                    well = entry.well

                    property_values: dict[str, np.ndarray] = {}
                    for rft_property in entry:
                        key = f"{well}{sep}{date}{sep}{rft_property}"
                        if matcher.fullmatch(key) is not None:
                            values = entry[rft_property]
                            if np.isdtype(values.dtype, np.float32):
                                property_values[rft_property] = values

                    if property_values:
                        valid_entry = RFTConfig.ValidRFTEntry(
                            property_values=property_values,
                            well_connection_cells=entry.connections,
                            filepath=filepath,
                            well=well,
                            date=date,
                        )
                        rft_data[well, date] = valid_entry

        except (FileNotFoundError, InvalidRFTError) as err:
            raise InvalidResponseFile(
                f"Could not read RFT from {filepath}: {err}"
            ) from err

        return rft_data

    @staticmethod
    def response_schema() -> dict[str, Any]:
        return {
            "response_key": pl.String,
            "well": pl.String,
            "date": pl.String,
            "property": pl.String,
            "time": pl.Date,
            "depth": pl.Float32,
            "values": pl.Float32,
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "cell_center": pl.Array(pl.Float32, 3),
            "cell_zones": pl.List(pl.String),
        }

    def _warn_about_missing_rft_responses(
        self,
        rft_data: dict[tuple[WellName, datetime.date], RFTConfig.ValidRFTEntry],
        rft_filename: str,
    ) -> None:
        well_time_keys = {
            (well, time)
            for well, time_dict in self.data_to_read.items()
            for time in time_dict
        }
        well_time_keys_rft_data = {(well, time.isoformat()) for well, time in rft_data}
        well_time_without_response: set[tuple[str, str]] = (
            well_time_keys - well_time_keys_rft_data
        )

        formatted_items = [
            f"{well}: {time}" for well, time in sorted(well_time_without_response)
        ]
        _warn_about_missing_responses(
            formatted_items, "well(s) at time(s)", rft_filename
        )

    def read_from_file(self, run_path: str, iens: int, iter_: int) -> pl.DataFrame:
        """Reads the RFT values from <RUNPATH>/<ECLBASE>.RFT"""
        if not self.data_to_read:
            return pl.DataFrame(schema=self.response_schema())

        rft_filepath = self._rft_filepath(self.input_files[0], run_path, iens, iter_)
        rft_data: dict[tuple[WellName, datetime.date], RFTConfig.ValidRFTEntry]
        rft_data = self._scan_rft(rft_filepath)

        self._warn_about_missing_rft_responses(rft_data, Path(rft_filepath).name)

        if not rft_data:
            return pl.DataFrame(schema=self.response_schema())

        egrid_filepath = self._ergrid_filepath(rft_filepath)
        grid = _read_egrid(egrid_filepath)

        def _cell_center(i: int, j: int, k: int) -> npt.NDArray[np.float32]:
            try:
                return grid.cell_corners(i - 1, j - 1, k - 1).mean(axis=0)
            except IndexError as err:
                raise InvalidResponseFile(
                    f"Grid coordinates ({i}, {j}, {k}) are out of bounds "
                    f"for grid file {egrid_filepath}"
                ) from err

        def _get_cell_center() -> pl.Expr:
            return pl.struct(["well_connection_cell"]).map_elements(
                lambda x: _cell_center(
                    x["well_connection_cell"][0],
                    x["well_connection_cell"][1],
                    x["well_connection_cell"][2],
                ),
                return_dtype=pl.Array(pl.Float32, 3),
            )

        def _get_cell_zone() -> pl.Expr:
            if self.zonemap:
                zonemap_path = self._zonemap_filepath(
                    self.zonemap, run_path, iens, iter_
                )
                zonemap = _get_zonemap(zonemap_path)
            else:
                zonemap = {}
            return (
                pl.col("well_connection_cell")
                .arr.get(2)
                .replace_strict(zonemap, default=None, return_dtype=pl.List(pl.String))
            )

        get_cell_column = _get_cell_center()
        get_cell_zone = _get_cell_zone()
        try:
            df = pl.concat(
                [
                    pl.DataFrame(
                        {
                            "response_key": [f"{well}:{time.isoformat()}:{prop}"],
                            "well": [well],
                            "date": [time.isoformat()],
                            "property": [prop],
                            "time": [time],
                            "depth": [rft_data[well, time].property_values["DEPTH"]],
                            "values": [vals],
                            "well_connection_cell": pl.Series(
                                [rft_data[well, time].well_connection_cells.tolist()],
                                dtype=pl.List(pl.Array(pl.Int64, 3)),
                            ),
                        }
                    )
                    .explode("depth", "values", "well_connection_cell")
                    .with_columns(
                        get_cell_column.alias("cell_center"),
                        get_cell_zone.alias("cell_zones"),
                    )
                    for (well, time), inner_dict in rft_data.items()
                    for prop, vals in inner_dict.property_values.items()
                    if prop != "DEPTH" and len(vals) > 0
                ]
            )
            return df.pipe(self._assert_schema, self.response_schema())
        except KeyError as err:
            raise InvalidResponseFile(
                f"Could not find {err.args[0]} in RFTFile {rft_filepath}"
            ) from err

    @staticmethod
    def location_metadata_schema() -> dict[str, Any]:
        return {
            "east": pl.Float32,
            "north": pl.Float32,
            "tvd": pl.Float32,
            "actual_zones": pl.List(pl.String),
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "well_connection_cell_center": pl.Array(pl.Float32, 3),
        }

    def obtain_location_metadata(
        self,
        run_path: str,
        iens: int,
        iter_: int,
        observations: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Obtains location metadata for the observations from provided simulation run.
        'east', 'north', and 'tvd' make a unique location. 'actual_zones' is a list of
        zones that the location belongs to according to the zonemap, and
        'well_connection_cell' is the grid cell that the location belongs to in current
        simulation.
        """
        locations: list[Point] = []
        for row in observations.iter_rows(named=True):
            location = (row["east"], row["north"], row["tvd"])
            if location not in locations:
                locations.append(location)

        rft_filepath = self._rft_filepath(self.input_files[0], run_path, iens, iter_)
        grid_filepath = self._ergrid_filepath(rft_filepath)

        if self.zonemap:
            zonemap_path = self._zonemap_filepath(self.zonemap, run_path, iens, iter_)
            zonemap = _get_zonemap(zonemap_path)
        else:
            zonemap = {}

        location_cell_map = self._map_locations_to_cells(grid_filepath, locations)

        def _get_zone_list(loc: Point) -> list[str]:
            grid_index = location_cell_map[loc].grid_index
            if grid_index is None:
                return []
            return zonemap.get(grid_index[-1], [])

        return pl.DataFrame(
            {
                "east": [loc[0] for loc in locations],
                "north": [loc[1] for loc in locations],
                "tvd": [loc[2] for loc in locations],
                "actual_zones": [_get_zone_list(loc) for loc in locations],
                "well_connection_cell": [
                    location_cell_map[loc].grid_index for loc in locations
                ],
                "well_connection_cell_center": [
                    location_cell_map[loc].cell_center for loc in locations
                ],
            },
            schema=self.location_metadata_schema(),
        )

    @staticmethod
    def enrich_observations_with_metadata(
        observations: pl.DataFrame,
        location_metadata: pl.DataFrame,
    ) -> pl.DataFrame:
        return observations.join(
            location_metadata,
            on=["east", "north", "tvd"],
            how="left",
        ).with_columns(
            [
                pl.col("zone").alias("expected_zone"),
            ]
        )

    @staticmethod
    def is_zone_valid() -> pl.Expr:
        return pl.col("expected_zone").is_null() | pl.col("expected_zone").is_in(
            pl.col("actual_zones")
        )

    @staticmethod
    def approximate_missing_rft_responses(
        responses: pl.LazyFrame,
        observations: pl.DataFrame,
    ) -> pl.LazyFrame:
        """Attempts to fill missing response values by interpolation or extrapolation
        from nearby points

        In some cases rft responses may be missing for well connection points, e.g.
        due to inactive cells. For each such point, missing a response, this function
        finds the two nearest responses, within the same zone, whose connecting segment
        contains the projection of the missing point, and estimates the missing value by
        linear interpolation. Falls back to extrapolation from the two nearest responses
        if no pair of responses allowing for interpolation is found.
        """

        responses_df = responses.filter(
            pl.col("values").is_not_nan() & pl.col("values").is_not_null()
        ).collect()

        if (
            # Only proceed if the response schema contains the necessary metadata for
            # interpolation/extrapolation. These columns may be missing in legacy
            # responses. Once all stored responses are guaranteed to include them, this
            # guard can be removed.
            "cell_center" not in responses_df.schema
            or "cell_zones" not in responses_df.schema
        ):
            return responses

        observations_with_missing_response = observations.join(
            responses_df,
            on=["response_key", "well_connection_cell"],
            how="anti",
        )

        if observations_with_missing_response.is_empty():
            return responses

        def _find_best_pair_for_value_approximation(
            point_to_approximate: tuple[float, float, float],
            points_with_responses: npt.NDArray[np.float32],
        ) -> tuple[int, int, float] | None:
            """
            Finds the two nearest points in points_with_responses to
            point_to_approximate so that interpolation can be performed. If no such pair
            exists it falls back to the best pair for extrapolation. If there are
            not at least two non-overlapping points in points_with_responses,
            returns None.

            Returns (ia, ib, t) where ia and ib are indices of the chosen point pair in
            points_with_responses and t is the scalar projection of point_to_approximate
            onto the segment from points_with_responses[ia] to points_with_responses[ib]
            (t=0 at ia, t=1 at ib). t in [0, 1] means interpolation;
            t outside that range means extrapolation.

            Note: candidate points are sorted by their individual distance to
            point_to_approximate and pairs are tried in that order (nearest–second,
            nearest–third, …). This is simple and works well for the typical case where
            well paths are quite straight without sharp curves.

            Alternatives such as ranking pairs by perpendicular distance from
            the missing point to the segment, or by the sum of both point distances were
            considered but not adopted.

            The assumption is that the approach of simply sorting individual points by
            distance, instead of sorting point-pairs by distance, will quickly find
            a good pair of points for interpolation if such a pair exists without having
            to check every combination of point-pairs.
            """
            dists = np.linalg.norm(points_with_responses - point_to_approximate, axis=1)
            order = np.argsort(dists)
            best_pair = None
            for a_idx in range(len(order)):
                for b_idx in range(a_idx + 1, len(order)):
                    ia, ib = int(order[a_idx]), int(order[b_idx])
                    seg = points_with_responses[ib] - points_with_responses[ia]
                    seg_len_sq = float(np.dot(seg, seg))
                    if seg_len_sq > 0:
                        t = (
                            float(
                                np.dot(
                                    point_to_approximate - points_with_responses[ia],
                                    seg,
                                )
                            )
                            / seg_len_sq
                        )
                        if 0.0 <= t <= 1.0:
                            return (ia, ib, t)
                        if best_pair is None:
                            # This is the best pair for extrapolation if no
                            # interpolating pair is found
                            best_pair = (ia, ib, t)
            return best_pair

        approximated_rows: list[dict[str, Any]] = []

        # Since there can be multiple observations at the same location with different
        # properties (e.g. pressure, swat, sgas), we might repeat the search for the
        # same pair of points for approximation multiple times since we handle missing
        # responses per response_key, which includes the property. However, handling
        # missing responses based on response_key is convenient and keeps complexity
        # down.
        for missing in observations_with_missing_response.iter_rows(named=True):
            # Use the cell center for approximation if available since responses
            # are associated with cells. Fall back to the east, north, tvd coordinates
            # of the observation if the cell center is not available due to legacy
            # location metadata.
            missing_response_point = missing.get(
                "well_connection_cell_center",
                (
                    missing["east"],
                    missing["north"],
                    missing["tvd"],
                ),
            )

            # Find candidate points for interpolation/extrapolation within the same
            # zone.
            candidate_points = responses_df.filter(
                (pl.col("response_key") == missing["response_key"])
                & pl.col("cell_zones").list.contains(missing["zone"])
            )
            if len(candidate_points) < 2:
                continue
            centers = candidate_points.get_column("cell_center").to_numpy()
            best_pair = _find_best_pair_for_value_approximation(
                missing_response_point, centers
            )

            if best_pair:
                ia, ib, t = best_pair
                values = (
                    candidate_points.get_column("values").gather([ia, ib]).to_list()
                )
                v0, v1 = values[0], values[1]

                template = candidate_points.row(ia, named=True)

                approximated_rows.append(
                    {
                        **template,
                        "depth": float(missing_response_point[2]),
                        "values": np.float32(v0 + t * (v1 - v0)),
                        "well_connection_cell": missing["well_connection_cell"],
                        "cell_center": [np.nan, np.nan, np.nan],
                        "cell_zones": [missing["zone"]],
                    }
                )

        if approximated_rows:
            responses = pl.concat(
                [
                    responses_df,
                    pl.DataFrame(approximated_rows, schema=responses_df.schema),
                ]
            ).lazy()
        return responses

    @property
    def response_type(self) -> str:
        return "rft"

    @property
    def match_key(self) -> list[str]:
        return ["well_connection_cell"]

    @property
    def index_key(self) -> list[str]:
        return ["east", "north", "tvd", "zone"]

    @override
    def match_key_dict_expr(self) -> pl.Expr:
        assert len(self.match_key) == 1
        col = self.match_key[0]

        well_connection_cell_to_str_expr = pl.concat_str(
            [
                pl.lit("["),
                pl.col(col)
                .cast(pl.Array(pl.String, 3))
                .arr.join(", ")
                .fill_null("None"),
                pl.lit("]"),
            ],
            separator="",
        )

        return pl.concat_str(
            pl.lit(f"{col}="),
            pl.when(pl.col(col).is_null())
            .then(pl.lit("None"))
            .otherwise(well_connection_cell_to_str_expr),
        )

    @classmethod
    def from_config_dict(cls, config_dict: ConfigDict) -> RFTConfig | None:
        if rfts := config_dict.get(ConfigKeys.RFT, []):
            eclbase: str | None = config_dict.get("ECLBASE")
            if eclbase is None:
                raise ConfigValidationError(
                    "In order to use rft responses, ECLBASE has to be set."
                )
            fm_steps = config_dict.get(ConfigKeys.FORWARD_MODEL, [])
            names = [fm_step[0] for fm_step in fm_steps]
            simulation_step_exists = any(
                any(sim in name.lower() for sim in ["eclipse", "flow"])
                for name in names
            )
            if not simulation_step_exists:
                ConfigWarning.warn(
                    "Config contains a RFT key but no forward model "
                    "step known to generate rft files"
                )

            declared_data: dict[WellName, dict[datetime.date, list[RFTProperty]]] = (
                defaultdict(lambda: defaultdict(list))
            )
            for rft in rfts:
                for expected in ["WELL", "DATE", "PROPERTIES"]:
                    if expected not in rft:
                        raise ConfigValidationError.with_context(
                            f"For RFT keyword {expected} must be specified.", rft
                        )
                well = rft["WELL"]
                props = [p.strip() for p in rft["PROPERTIES"].split(",")]
                time = rft["DATE"]
                declared_data[well][time] += props
            data_to_read = {
                well: {time: sorted(set(p)) for time, p in inner_dict.items()}
                for well, inner_dict in declared_data.items()
            }
            keys = sorted(
                {
                    f"{well}:{time}:{p}"
                    for well, inner_dict in declared_data.items()
                    for time, props in inner_dict.items()
                    for p in props
                }
            )

            return cls(
                name="rft",
                input_files=[eclbase.replace("%d", "<IENS>")],
                keys=keys,
                data_to_read=data_to_read,
                zonemap=config_dict.get(ConfigKeys.ZONEMAP),
                approximate_missing_values=config_dict.get(
                    ConfigKeys.APPROXIMATE_MISSING_RFT_VALUES, False
                ),
            )

        return None


responses_index.add_response_type(RFTConfig)
