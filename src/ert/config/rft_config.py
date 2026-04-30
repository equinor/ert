from __future__ import annotations

import datetime
import fnmatch
import logging
import re
import warnings
from collections import defaultdict
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

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
from ert.warnings import PostExperimentWarning

from .parsing import (
    ConfigDict,
    ConfigKeys,
    ConfigValidationError,
    ConfigWarning,
    parse_zonemap,
)
from .response_config import InvalidResponseFile, ResponseConfig
from .responses_index import responses_index

logger = logging.getLogger(__name__)


# A Point in UTM/TVD coordinates
Point: TypeAlias = tuple[float, float, float]
# Index to a cell in a grid
GridIndex: TypeAlias = tuple[int, int, int]
ZoneName: TypeAlias = str
WellName: TypeAlias = str
DateString: TypeAlias = str
RFTProperty: TypeAlias = str


@dataclass(frozen=True)
class WellPoint:
    """A well path point optionally constrained to be in a given zone."""

    well_name: str | None = None
    point: tuple[float | None, float | None, float | None] = (None, None, None)
    zone_name: ZoneName | None = None

    def has_zone(self) -> bool:
        return self.zone_name is not None


class RFTConfig(ResponseConfig):
    """:term:`RFT` response from a :term:`reservoir simulator`.

    RFTConfig is the configuration of responses in the <RUNPATH>/<ECLBASE>.RFT
    file which may be generated from a reservoir simulator forward model step.

    The file contains values for grid cells along a wellpath (see RFTReader for
    details). RFTConfig will match the values against the given :term:`UTM`/:term:`TVD`
    locations.

    Parameters:
        data_to_read: dictionary of the values that should be read from the rft file.
        well_locations: list of optionally zone constrained well points that the rft
        values should be labeled with.
        zonemap: The mapping from grid layer index to zone name.
        interpolate_missing_values: option to fill missing values by interpolating
            between points within same zone.
    """

    type: Literal["rft"] = "rft"
    name: str = "rft"
    has_finalized_keys: bool = False
    data_to_read: dict[WellName, dict[DateString, list[RFTProperty]]] = Field(
        default_factory=dict
    )
    well_locations: list[WellPoint] = Field(default_factory=list)
    zonemap: Path | None = None
    interpolate_missing_values: bool = False

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

    def _read_grid(self, egrid_file: str) -> CornerpointGrid:
        try:
            grid = CornerpointGrid.read_egrid(egrid_file)
        except (OSError, InvalidEgridFileError) as err:
            raise InvalidResponseFile(f"Could not read grid file: {err}") from err
        return grid

    @staticmethod
    def _map_locations_to_cells(
        grid: CornerpointGrid | None, zoned_locations: list[WellPoint]
    ) -> dict[WellPoint, GridIndex]:
        """
        For each location, find the corresponding connected grid cell, if it exists.
        """

        location_cell_map: dict[WellPoint, GridIndex] = {}
        if not zoned_locations:
            return location_cell_map

        if grid is None:
            raise InvalidResponseFile(
                "RFTConfig is configured with well locations but "
                "no grid could be read to map those locations to cells."
            )
        for zoned_location, cell in zip(
            zoned_locations,
            grid.find_cell_containing_point(
                [cast(Point, loc.point) for loc in zoned_locations]
            ),
            strict=True,
        ):
            if cell is None:
                raise InvalidResponseFile(
                    f"Did not find grid coordinate for location(s) {zoned_location}"
                )
            # cells returned by grid are 0-based, while zonemap
            # and RFTEntry.connections are 1-based, so unifying
            location_cell_map[zoned_location] = (cell[0] + 1, cell[1] + 1, cell[2] + 1)
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
            else:
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
        try:
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

    def read_from_file(self, run_path: str, iens: int, iter_: int) -> pl.DataFrame:
        """Reads the RFT values from <RUNPATH>/<ECLBASE>.RFT

        Also labels those values by which optionally zone constrained point
        it belongs to.

        The columns east, north, tvd is none when the value does not belong to
        any point, otherwise it is the x,y,z values of that point. If the point
        is constrained to be in a certain zone then the zone column is also populated.

        Points which were constrained to be in a given zone, but were not contained
        in that zone, is not labeled, and instead a warning is emitted.
        """
        schema: dict[str, Any] = {
            "response_key": pl.String,
            "well": pl.String,
            "date": pl.String,
            "property": pl.String,
            "time": pl.Date,
            "depth": pl.Float32,
            "values": pl.Float32,
            "zone": pl.String,
            "east": pl.Float32,
            "north": pl.Float32,
            "tvd": pl.Float32,
            "i": pl.Int64,
            "j": pl.Int64,
            "k": pl.Int64,
        }

        if not self.data_to_read:
            return pl.DataFrame(schema=schema)

        rft_filepath = self._rft_filepath(self.input_files[0], run_path, iens, iter_)
        rft_data: dict[tuple[WellName, datetime.date], RFTConfig.ValidRFTEntry]
        rft_data = self._scan_rft(rft_filepath)

        if not rft_data:
            return pl.DataFrame(schema=schema)

        grid_filepath = self._ergrid_filepath(rft_filepath)
        grid = self._read_grid(grid_filepath) if self.well_locations else None
        location_metadata = self._obtain_location_metadata(grid, run_path, iens, iter_)

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
                    ).explode("depth", "values", "well_connection_cell")
                    for (well, time), inner_dict in rft_data.items()
                    for prop, vals in inner_dict.property_values.items()
                    if prop != "DEPTH" and len(vals) > 0
                ]
            )
        except KeyError as err:
            raise InvalidResponseFile(
                f"Could not find {err.args[0]} in RFTFile {rft_filepath}"
            ) from err

        if self.zonemap:
            zonemap_path = self._zonemap_filepath(self.zonemap, run_path, iens, iter_)
            zonemap = parse_zonemap(str(zonemap_path), zonemap_path.read_text())
        else:
            zonemap = {}

        df = df.with_columns(
            pl.col("well_connection_cell")
            .arr.get(2)
            .replace_strict(zonemap, default=None, return_dtype=pl.List(pl.String))
            .alias("actual_zones")
        )

        combined = self._combine_response_and_location_metadata(
            df, location_metadata, iens, iter_
        )

        if self.interpolate_missing_values:
            if grid is not None:
                combined = self._approximate_missing_rft_responses(
                    combined, location_metadata, grid
                )
            else:
                warnings.warn(
                    ConfigWarning(
                        "RFTConfig is configured to interpolate missing values but no "
                        "grid could be read. Missing values will not be interpolated."
                    ),
                    stacklevel=2,
                )

        return combined.drop("well_name", "actual_zones").pipe(
            self._assert_schema, schema
        )

    def _obtain_location_metadata(
        self,
        grid: CornerpointGrid | None,
        run_path: str,
        iens: int,
        iter_: int,
    ) -> pl.DataFrame:
        """
        Obtains location metadata for the observations from provided simulation run.
        'location' and 'expected_zone' is data provided by the observations. 'location'
        is a primary key, while 'expected_zone' is added for completeness.
        'actual_zones' is a list of zones that the location belongs to according to the
        zonemap, and 'actual_cell' is the grid cell that the location belongs to in
        current simulation.
        """

        location_cell_map = self._map_locations_to_cells(grid, self.well_locations)

        return pl.DataFrame(
            {
                "location": pl.Series(
                    [loc.point for loc in self.well_locations],
                    dtype=pl.Array(pl.Float32, 3),
                ),
                "expected_zone": pl.Series(
                    [loc.zone_name for loc in self.well_locations], dtype=pl.String
                ),
                "actual_cell": pl.Series(
                    [location_cell_map[loc] for loc in self.well_locations],
                    dtype=pl.Array(pl.Int64, 3),
                ),
                "well_name": pl.Series(
                    [loc.well_name for loc in self.well_locations], dtype=pl.String
                ),
            }
        )

    @staticmethod
    def _approximate_missing_rft_responses(
        responses: pl.DataFrame,
        location_metadata: pl.DataFrame,
        grid: CornerpointGrid,
    ) -> pl.DataFrame:
        """Attempts to fill missing response values by interpolation or extrapolation
        from nearby points

        In some cases rft responses may be missing for well connection points,  e.g.
        due to inactive cells. This function projects the point missing a response onto
        the line through the two nearest RFT responses that allows for interpolation
        within the same zone, using cell centers computed from the EGRID.
        Falls back to extrapolation from the two nearest responses if no pair of points
        allowing for interpolation is found.
        """
        locations_with_missing_response = location_metadata.with_columns(
            pl.col("actual_cell").arr.get(0).alias("i"),
            pl.col("actual_cell").arr.get(1).alias("j"),
            pl.col("actual_cell").arr.get(2).alias("k"),
        ).join(
            responses,
            left_on=["well_name", "i", "j", "k"],
            right_on=["well", "i", "j", "k"],
            how="anti",
        )

        if not locations_with_missing_response.is_empty():

            def _find_best_pair_for_value_approximation(
                point_to_approximate: tuple[float, float, float],
                points_with_responses: npt.NDArray[np.float32],
            ) -> tuple[int, int, float] | None:
                """
                Finds the two nearest points among points_with_responses to the
                point_to_approximate that allows for interpolation. If no such pair
                exists it falls back to the best pair for extrapolation. If there are
                not at least two non overlapping points in points_with_responses,
                returns None.
                """
                dists = np.linalg.norm(
                    points_with_responses - point_to_approximate, axis=1
                )
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
                                        point_to_approximate
                                        - points_with_responses[ia],
                                        seg,
                                    )
                                )
                                / seg_len_sq
                            )
                            if 0.0 <= t <= 1.0:
                                return (ia, ib, t)
                            elif best_pair is None:
                                # This is the best pair for extrapolation if no
                                # interpolating pair is found
                                best_pair = (ia, ib, t)
                return best_pair

            def _cell_center(i: int, j: int, k: int) -> npt.NDArray[np.float32]:
                return grid.cell_corners(i - 1, j - 1, k - 1).mean(axis=0)

            approximated_rows: list[dict[str, Any]] = []

            for missing in locations_with_missing_response.iter_rows(named=True):
                missing_response_point = (
                    missing["location"][0],
                    missing["location"][1],
                    missing["location"][2],
                )

                # Find candidate points for interpolation/extrapolation within the same
                # zone and belonging to the same well.
                candidate_points = responses.filter(
                    (
                        (
                            pl.col("zone").is_not_null()
                            & (pl.col("zone") == missing["expected_zone"])
                        )
                        | (
                            pl.col("zone").is_null()
                            & pl.col("actual_zones").list.contains(
                                missing["expected_zone"]
                            )
                        )
                    )
                    & (pl.col("well") == missing["well_name"])
                )
                response_keys = candidate_points["response_key"].unique().to_list()
                for rk in response_keys:
                    date_and_prop_candidates = candidate_points.filter(
                        pl.col("response_key") == rk
                    )
                    if len(date_and_prop_candidates) >= 2:
                        # Compute cell centers for all candidate response rows
                        centers = np.array(
                            [
                                _cell_center(
                                    int(row["i"]), int(row["j"]), int(row["k"])
                                )
                                for row in date_and_prop_candidates.iter_rows(
                                    named=True
                                )
                            ],
                            dtype=np.float32,
                        )

                        best_pair = _find_best_pair_for_value_approximation(
                            missing_response_point, centers
                        )

                        if best_pair:
                            ia, ib, t = best_pair
                            v0 = float(date_and_prop_candidates["values"][ia])
                            v1 = float(date_and_prop_candidates["values"][ib])

                            template = date_and_prop_candidates.row(ia, named=True)
                            approximated_rows.append(
                                {
                                    **template,
                                    "depth": float(missing_response_point[2]),
                                    "values": np.float32(v0 + t * (v1 - v0)),
                                    "east": float(missing_response_point[0]),
                                    "north": float(missing_response_point[1]),
                                    "tvd": float(missing_response_point[2]),
                                    "i": missing["actual_cell"][0],
                                    "j": missing["actual_cell"][1],
                                    "k": missing["actual_cell"][2],
                                }
                            )

            if approximated_rows:
                responses = pl.concat(
                    [
                        responses,
                        pl.DataFrame(approximated_rows, schema=responses.schema),
                    ]
                )
        return responses

    @staticmethod
    def _combine_response_and_location_metadata(
        responses: pl.DataFrame,
        location_metadata: pl.DataFrame,
        iens: int,
        iter_: int,
    ) -> pl.DataFrame:
        result = responses.join(
            location_metadata,
            left_on="well_connection_cell",
            right_on="actual_cell",
            how="left",
        )

        is_zone_valid = pl.col("expected_zone").is_null() | pl.col(
            "expected_zone"
        ).is_in(pl.col("actual_zones"))

        disabled_due_to_zone_mismatch = result.filter(
            pl.col("expected_zone").is_not_null() & ~is_zone_valid
        )
        for row in disabled_due_to_zone_mismatch.iter_rows(named=True):
            warnings.warn(
                PostExperimentWarning(
                    f"An RFT observation with location {row['location']}, "
                    f"in iteration {iter_}, realization {iens} did "
                    f"not match expected zone {row['expected_zone']}. The observation "
                    "was deactivated",
                ),
                stacklevel=2,
            )
        result = result.filter(is_zone_valid)

        def location_to_coordinate(index: int, name: str) -> pl.Expr:
            return (
                pl.when(pl.col("location").is_null())
                .then(None)
                .otherwise(pl.col("location").arr.get(index))
                .alias(name)
            )

        return result.with_columns(
            [
                pl.col("expected_zone").alias("zone"),
                location_to_coordinate(0, "east"),
                location_to_coordinate(1, "north"),
                location_to_coordinate(2, "tvd"),
                pl.col("well_connection_cell").arr.get(0).alias("i"),
                pl.col("well_connection_cell").arr.get(1).alias("j"),
                pl.col("well_connection_cell").arr.get(2).alias("k"),
            ]
        ).drop(
            [
                "location",
                "well_connection_cell",
                "expected_zone",
            ]
        )

    @property
    def response_type(self) -> str:
        return "rft"

    @property
    def match_key(self) -> list[str]:
        return ["east", "north", "tvd", "zone"]

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
                interpolate_missing_values=config_dict.get(
                    ConfigKeys.INTERPOLATE_RFT_VALUES, False
                ),
            )

        return None


responses_index.add_response_type(RFTConfig)
