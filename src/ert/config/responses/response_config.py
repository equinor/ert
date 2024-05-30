import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import xarray as xr

from ert.config.commons import Refcase
from ert.config.parameter_config import CustomDict
from ert.config.responses.observation_vector import ObsVector


@dataclasses.dataclass
class ObsArgs:
    obs_name: str
    refcase: Optional[Refcase]
    values: Any
    std_cutoff: Any
    history: Optional[Any]
    obs_time_list: Any
    config_for_response: Optional["ResponseConfig"] = None


@dataclasses.dataclass
class ResponseConfig(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def parse_observation(args: ObsArgs) -> Dict[str, ObsVector]: ...

    @abstractmethod
    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset: ...

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self, dict_factory=CustomDict)
        data["_ert_kind"] = self.__class__.__name__
        return data
