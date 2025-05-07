from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, no_type_check

import polars as pl

from ert.substitutions import substitute_runpath_name

from ._read_summary import read_summary
from .ensemble_config import Refcase
from .parsing import ConfigDict, ConfigKeys
from .parsing.config_errors import ConfigValidationError, ConfigWarning
from .response_config import InvalidResponseFile, ResponseConfig, ResponseMetadata
from .responses_index import responses_index

logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig(ResponseConfig):
    name: str = "summary"
    refcase: set[datetime] | list[str] | None = None
    has_finalized_keys: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.refcase, list):
            self.refcase = {datetime.fromisoformat(val) for val in self.refcase}
        self.keys = sorted(set(self.keys))
        if len(self.keys) < 1:
            raise ValueError("SummaryConfig must be given at least one key")

    @property
    def metadata(self) -> list[ResponseMetadata]:
        return [
            ResponseMetadata(
                response_type=self.name,
                response_key=response_key,
                filter_on=None,
            )
            for response_key in self.keys
        ]

    @property
    def expected_input_files(self) -> list[str]:
        base = self.input_files[0]
        return [f"{base}.UNSMRY", f"{base}.SMSPEC"]

    def read_from_file(self, run_path: str, iens: int, iter_: int) -> pl.DataFrame:
        filename = substitute_runpath_name(self.input_files[0], iens, iter_)
        _, keys, time_map, data = read_summary(f"{run_path}/{filename}", self.keys)
        if len(data) == 0 or len(keys) == 0:
            # https://github.com/equinor/ert/issues/6974
            # There is a bug with storing empty responses so we have
            # to raise an error in that case
            raise InvalidResponseFile(
                f"Did not find any summary values matching {self.keys} in {filename}"
            )

        # Important: Pick lowest unit resolution to allow for using
        # datetimes many years into the future
        time_map_series = pl.Series(time_map).dt.cast_time_unit("ms")
        df = pl.DataFrame(
            {
                "response_key": keys,
                "time": [time_map_series for _ in data],
                "values": [pl.Series(row, dtype=pl.Float32) for row in data],
            }
        )
        df = df.explode("values", "time")
        df = df.sort(by=["time"])
        return df

    @property
    def response_type(self) -> str:
        return "summary"

    @property
    def primary_key(self) -> list[str]:
        return ["time"]

    @no_type_check
    @classmethod
    def from_config_dict(cls, config_dict: ConfigDict) -> SummaryConfig | None:
        refcase = Refcase.from_config_dict(config_dict)
        if summary_keys := config_dict.get(ConfigKeys.SUMMARY, []):
            eclbase: str | None = config_dict.get("ECLBASE")
            if eclbase is None:
                raise ConfigValidationError(
                    "In order to use summary responses, ECLBASE has to be set."
                )
            time_map = set(refcase.dates) if refcase is not None else None
            fm_steps = config_dict.get(ConfigKeys.FORWARD_MODEL, [])
            names = [fm_step[0] for fm_step in fm_steps]
            simulation_step_exists = any(
                any(sim in _name.lower() for sim in ["eclipse", "flow"])
                for _name in names
            )
            if not simulation_step_exists:
                ConfigWarning.warn(
                    "Config contains a SUMMARY key but no forward model "
                    "steps known to generate a summary file"
                )
            return cls(
                name="summary",
                input_files=[eclbase.replace("%d", "<IENS>")],
                keys=[key for keys in summary_keys for key in keys],
                refcase=time_map,
            )

        return None

    @classmethod
    def display_column(cls, value: Any, column_name: str) -> str:
        if column_name == "time":
            return value.strftime("%Y-%m-%d")

        return str(value)


responses_index.add_response_type(SummaryConfig)
