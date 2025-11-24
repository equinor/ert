from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from typing import Any, Literal

import numpy.typing as npt
import polars as pl
from pydantic import Field
from resfo_utilities import RFTReader

from ert.substitutions import substitute_runpath_name

from .parsing import ConfigDict, ConfigKeys, ConfigValidationError, ConfigWarning
from .response_config import ResponseConfig, ResponseMetadata
from .responses_index import responses_index

logger = logging.getLogger(__name__)


class RFTConfig(ResponseConfig):
    type: Literal["rft"] = "rft"
    name: str = "rft"
    has_finalized_keys: bool = False
    data_to_read: dict[str, dict[date, list[str]]] = Field(default_factory=dict)

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
        return [f"{base}.RFT"]

    def read_from_file(self, run_path: str, iens: int, iter_: int) -> pl.DataFrame:
        filename = substitute_runpath_name(self.input_files[0], iens, iter_)
        fetched: dict[tuple[str, date], dict[str, npt.NDArray[Any]]] = defaultdict(dict)
        with RFTReader.open(f"{run_path}/{filename}") as rft:
            for entry in rft:
                key = (entry.well, entry.date)
                to_get = self.data_to_read.get(entry.well, {}).get(entry.date, [])
                if to_get and "DEPTH" not in to_get:
                    to_get.append("DEPTH")
                for t in to_get:
                    if t in entry:
                        fetched[key][t] = entry[t]

        return pl.concat(
            [
                pl.DataFrame(
                    {
                        "response_key": [f"{well}:{time.isoformat()}:{prop}"],
                        "time": [time],
                        "depth": [fetched[well, time]["DEPTH"]],
                        "values": [vals],
                    }
                )
                for (well, time), inner_dict in fetched.items()
                for prop, vals in inner_dict.items()
                if prop != "DEPTH"
            ]
        )

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
                    "steps known to generate rft files"
                )

            declared_data: dict[str, dict[date, list[str]]] = defaultdict(
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
                time = date.fromisoformat(rft["DATE"])
                declared_data[well][time] += props
            data_to_read = {
                well: {time: sorted(set(p)) for time, p in inner_dict.items()}
                for well, inner_dict in declared_data.items()
            }
            keys = sorted(
                {
                    f"{well}:{time.isoformat()}:{p}"
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
