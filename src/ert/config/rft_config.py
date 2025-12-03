from __future__ import annotations

import datetime
import fnmatch
import logging
import re
from collections import defaultdict
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import Field
from resfo_utilities import InvalidRFTError, RFTReader

from ert.substitutions import substitute_runpath_name

from .parsing import ConfigDict, ConfigKeys, ConfigValidationError, ConfigWarning
from .response_config import InvalidResponseFile, ResponseConfig, ResponseMetadata
from .responses_index import responses_index

logger = logging.getLogger(__name__)


class RFTConfig(ResponseConfig):
    type: Literal["rft"] = "rft"
    name: str = "rft"
    has_finalized_keys: bool = False
    data_to_read: dict[str, dict[str, list[str]]] = Field(default_factory=dict)

    @property
    def metadata(self) -> list[ResponseMetadata]:
        return [
            ResponseMetadata(
                response_type=self.name,
                response_key=response_key,
                filter_on=None,
                finalized=self.has_finalized_keys,
            )
            for response_key in self.keys
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

    def read_from_file(self, run_path: str, iens: int, iter_: int) -> pl.DataFrame:
        filename = substitute_runpath_name(self.input_files[0], iens, iter_)
        if filename.upper().endswith(".DATA"):
            # For backwards compatibility, it is
            # allowed to give REFCASE and ECLBASE both
            # with and without .DATA extensions
            filename = filename[:-5]
        fetched: dict[tuple[str, datetime.date], dict[str, npt.NDArray[np.float32]]] = (
            defaultdict(dict)
        )
        location: dict[
            tuple[str, datetime.date], np.ndarray[tuple[int, int], np.dtype[np.int32]]
        ] = {}
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
        try:
            with RFTReader.open(f"{run_path}/{filename}") as rft:
                for entry in rft:
                    added_location = False
                    date = entry.date
                    well = entry.well
                    for rft_property in entry:
                        key = f"{well}{sep}{date}{sep}{rft_property}"
                        if matcher.fullmatch(key) is not None:
                            values = entry[rft_property]
                            if np.isdtype(values.dtype, np.float32):
                                fetched[well, date][rft_property] = values
                                if not added_location:
                                    location[well, date] = entry.connections
                                    added_location = True
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
                }
            )

        try:
            return pl.concat(
                [
                    pl.DataFrame(
                        {
                            "response_key": [f"{well}:{time.isoformat()}:{prop}"],
                            "time": [time],
                            "property": [prop],
                            "depth": [fetched[well, time]["DEPTH"]],
                            "grid_i": [location[well, time][:, 0]],
                            "grid_j": [location[well, time][:, 1]],
                            "grid_k": [location[well, time][:, 2]],
                            "values": [vals],
                        }
                    ).explode("depth", "values", "grid_i", "grid_j", "grid_k")
                    for (well, time), inner_dict in fetched.items()
                    for prop, vals in inner_dict.items()
                    if prop != "DEPTH"
                ]
            )
        except KeyError as err:
            raise InvalidResponseFile(
                f"Could not find {err.args[0]} in RFTFile {filename}"
            ) from err

    @property
    def response_type(self) -> str:
        return "rft"

    @property
    def primary_key(self) -> list[str]:
        return []

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
                any(sim in _name.lower() for sim in ["eclipse", "flow"])
                for _name in names
            )
            if not simulation_step_exists:
                ConfigWarning.warn(
                    "Config contains a RFT key but no forward model "
                    "step known to generate rft files"
                )

            declared_data: dict[str, dict[datetime.date, list[str]]] = defaultdict(
                lambda: defaultdict(list)
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
            )

        return None


responses_index.add_response_type(RFTConfig)
