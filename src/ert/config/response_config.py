import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import xarray as xr
from typing_extensions import Self

from ert.config.parameter_config import CustomDict
from ert.config.parsing import ConfigDict


@dataclasses.dataclass
class ResponseConfig(ABC):
    name: str
    input_files: List[str] = dataclasses.field(default_factory=list)
    keys: List[str] = dataclasses.field(default_factory=list)

    @abstractmethod
    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset: ...

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self, dict_factory=CustomDict)
        data["_ert_kind"] = self.__class__.__name__
        return data

    @property
    @abstractmethod
    def expected_input_files(self) -> List[str]:
        """Returns a list of filenames expected to be produced by the forward model"""

    @property
    @abstractmethod
    def response_type(self) -> str:
        """Label to identify what kind of response it is.
        Must not overlap with that of other response configs."""
        ...

    @classmethod
    @abstractmethod
    def from_config_dict(cls, config_dict: ConfigDict) -> Optional[Self]:
        """Creates a config, given an ert config dict.
        A response config may depend on several config kws, such as REFCASE
        for summary."""
