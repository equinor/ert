from __future__ import annotations

import datetime
import fnmatch
import logging
import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import IO, Any, Literal, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import Field
from resfo_utilities import CornerpointGrid, InvalidRFTError, RFTReader

from ert.substitutions import substitute_runpath_name
from ert.warnings import PostSimulationWarning

from .parsing import ConfigDict, ConfigKeys, ConfigValidationError, ConfigWarning
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
class _ZonedPoint:
    """A point optionally constrained to be in a given zone."""

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
        loations: list of optionally zone constrained points that the rft values should
            be labeled with.
        zonemap: The mapping from grid layer index to zone name.
    """

    type: Literal["rft"] = "rft"
    name: str = "rft"
    has_finalized_keys: bool = False
    data_to_read: dict[WellName, dict[DateString, list[RFTProperty]]] = Field(
        default_factory=dict
    )
    locations: list[Point | tuple[Point, ZoneName]] = Field(default_factory=list)
    zonemap: dict[int, list[ZoneName]] = Field(default_factory=dict)

    @property
    def _zoned_locations(self) -> list[_ZonedPoint]:
        return [
            _ZonedPoint(*p) if isinstance(p[1], ZoneName) else _ZonedPoint(p)
            for p in self.locations
        ]

    @property
    def expected_input_files(self) -> list[str]:
        base = self.input_files[0]
        if base.upper().endswith(".DATA"):
            # For backwards compatibility, it is
            # allowed to give REFCASE and ECLBASE both
            # with and without .DATA extensions
            base = base[:-5]

        return [f"{base}.RFT"]

    def _find_indices(
        self, egrid_file: str | os.PathLike[str] | IO[Any]
    ) -> dict[GridIndex | None, set[_ZonedPoint]]:
        indices = defaultdict(set)
        for a, b in zip(
            CornerpointGrid.read_egrid(egrid_file).find_cell_containing_point(
                [cast(Point, loc.point) for loc in self._zoned_locations]
            ),
            self._zoned_locations,
            strict=True,
        ):
            indices[a].add(b)
        return indices

    def _filter_zones(
        self,
        indices: dict[GridIndex | None, set[_ZonedPoint]],
        iens: int,
        iter_: int,
    ) -> dict[GridIndex | None, set[_ZonedPoint]]:
        for idx, locs in indices.items():
            if idx is not None:
                for loc in list(locs):
                    if loc.has_zone():
                        zone = cast(ZoneName, loc.zone_name)
                        # zonemap is 1-indexed so +1
                        if zone not in self.zonemap.get(idx[-1] + 1, []):
                            warnings.warn(
                                PostSimulationWarning(
                                    f"An RFT observation with location {loc.point}, "
                                    f"in iteration {iter_}, realization {iens} did "
                                    f"not match expected zone {zone}. The observation "
                                    "was deactivated",
                                ),
                                stacklevel=2,
                            )
                            locs.remove(loc)
        return indices

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
        filename = substitute_runpath_name(self.input_files[0], iens, iter_)
        if filename.upper().endswith(".DATA"):
            # For backwards compatibility, it is
            # allowed to give REFCASE and ECLBASE both
            # with and without .DATA extensions
            filename = filename[:-5]
        grid_filename = f"{run_path}/{filename}"
        if grid_filename.upper().endswith(".RFT"):
            grid_filename = grid_filename[:-4]
        grid_filename += ".EGRID"
        fetched: dict[
            tuple[WellName, datetime.date], dict[RFTProperty, npt.NDArray[np.float32]]
        ] = defaultdict(dict)
        indices = {}
        if self.locations:
            indices = self._filter_zones(self._find_indices(grid_filename), iens, iter_)
        if None in indices:
            raise InvalidResponseFile(
                f"Did not find grid coordinate for location(s) {indices[None]}"
            )
        # This is a somewhat complicated optimization in order to
        # support wildcards in well names, dates and properties
        # A python for loop is too slow so we use a compiled regex
        # instead
        if not self.data_to_read:
            return pl.DataFrame(
                {
                    "response_key": [],
                    "time": [],
                    "depth": [],
                    "values": [],
                    "east": [],
                    "north": [],
                    "tvd": [],
                    "zone": [],
                }
            )

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
        locations = {}
        try:
            with RFTReader.open(f"{run_path}/{filename}") as rft:
                for entry in rft:
                    date = entry.date
                    well = entry.well
                    for rft_property in entry:
                        key = f"{well}{sep}{date}{sep}{rft_property}"
                        if matcher.fullmatch(key) is not None:
                            values = entry[rft_property]
                            locations[well, date] = [
                                list(
                                    indices.get(
                                        (c[0] - 1, c[1] - 1, c[2] - 1),
                                        [_ZonedPoint()],
                                    )
                                )
                                for c in entry.connections
                            ]
                            if np.isdtype(values.dtype, np.float32):
                                fetched[well, date][rft_property] = values
        except (FileNotFoundError, InvalidRFTError) as err:
            raise InvalidResponseFile(
                f"Could not read RFT from {run_path}/{filename}: {err}"
            ) from err

        if not fetched:
            return pl.DataFrame(
                {
                    "response_key": [],
                    "time": [],
                    "depth": [],
                    "values": [],
                    "east": [],
                    "north": [],
                    "tvd": [],
                    "zone": [],
                }
            )

        try:
            df = pl.concat(
                [
                    pl.DataFrame(
                        {
                            "response_key": [f"{well}:{time.isoformat()}:{prop}"],
                            "time": [time],
                            "depth": [fetched[well, time]["DEPTH"]],
                            "values": [vals],
                            "location": pl.Series(
                                [
                                    [
                                        [loc.point for loc in locs]
                                        for locs in locations.get(
                                            (well, time),
                                            [[_ZonedPoint()]] * len(vals),
                                        )
                                    ]
                                ],
                                dtype=pl.Array(
                                    pl.List(pl.Array(pl.Float32, 3)), len(vals)
                                ),
                            ),
                            "zone": pl.Series(
                                [
                                    [
                                        [loc.zone_name for loc in locs]
                                        for locs in locations.get(
                                            (well, time),
                                            [[_ZonedPoint()]] * len(vals),
                                        )
                                    ]
                                ],
                                dtype=pl.Array(pl.List(pl.String), len(vals)),
                            ),
                        }
                    )
                    .explode("depth", "values", "location", "zone")
                    .explode("location", "zone")
                    for (well, time), inner_dict in fetched.items()
                    for prop, vals in inner_dict.items()
                    if prop != "DEPTH" and len(vals) > 0
                ]
            )
        except KeyError as err:
            raise InvalidResponseFile(
                f"Could not find {err.args[0]} in RFTFile {filename}"
            ) from err

        return df.with_columns(
            east=pl.col("location").arr.get(0),
            north=pl.col("location").arr.get(1),
            tvd=pl.col("location").arr.get(2),
        ).drop("location")

    @property
    def response_type(self) -> str:
        return "rft"

    @property
    def primary_key(self) -> list[str]:
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
                zonemap=config_dict.get(ConfigKeys.ZONEMAP, ("", {}))[1],
            )

        return None


responses_index.add_response_type(RFTConfig)
