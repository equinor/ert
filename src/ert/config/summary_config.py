from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Set, Union

import xarray as xr

from ._read_summary import read_summary
from .ensemble_config import Refcase
from .parsing import ConfigDict, ConfigKeys
from .parsing.config_errors import ConfigValidationError
from .response_config import ResponseConfig
from .responses_index import responses_index

if TYPE_CHECKING:
    from typing import List

logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig(ResponseConfig):
    name: str = "summary"
    refcase: Union[Set[datetime], List[str], None] = None

    def __post_init__(self) -> None:
        if isinstance(self.refcase, list):
            self.refcase = {datetime.fromisoformat(val) for val in self.refcase}
        self.keys = sorted(set(self.keys))
        if len(self.keys) < 1:
            raise ValueError("SummaryConfig must be given at least one key")

    @property
    def expected_input_files(self) -> List[str]:
        base = self.input_files[0]
        return [f"{base}.UNSMRY", f"{base}.SMSPEC"]

    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset:
        filename = self.input_files[0].replace("<IENS>", str(iens))
        _, keys, time_map, data = read_summary(f"{run_path}/{filename}", self.keys)
        if len(data) == 0 or len(keys) == 0:
            # https://github.com/equinor/ert/issues/6974
            # There is a bug with storing empty responses so we have
            # to raise an error in that case
            raise ValueError(
                f"Did not find any summary values matching {self.keys} in {filename}"
            )
        ds = xr.Dataset(
            {"values": (["name", "time"], data)},
            coords={"time": time_map, "name": keys},
        )
        return ds.drop_duplicates("time")

    @property
    def response_type(self) -> str:
        return "summary"

    @classmethod
    def from_config_dict(self, config_dict: ConfigDict) -> Optional[SummaryConfig]:
        refcase = Refcase.from_config_dict(config_dict)
        eclbase = config_dict.get("ECLBASE")  # type: ignore
        if eclbase is not None:
            eclbase = eclbase.replace("%d", "<IENS>")

        summary_keys = config_dict.get(ConfigKeys.SUMMARY, [])  # type: ignore
        if summary_keys:
            if eclbase is None:
                raise ConfigValidationError(
                    "In order to use summary responses, ECLBASE has to be set."
                )
            time_map = set(refcase.dates) if refcase is not None else None

            return SummaryConfig(
                name="summary",
                input_files=[eclbase],
                keys=[key for keys in summary_keys for key in keys],
                refcase=time_map,
            )

        return None


responses_index.add_response_type(SummaryConfig)
