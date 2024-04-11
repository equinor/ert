import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import xarray as xr

from ert.config.commons import CustomDict, Refcase

from .observation_vector import ObsVector
from .response_properties import ResponseDataInitialLayout


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

    @property
    @abstractmethod
    def primary_keys(self) -> List[str]:
        """
        Represents primary keys used to identify unique entries in the response.
        Does not include "name" which is the name of the
        individual response of this type.
        For example, for summary it is "time", and for
        gen_data it is ("index", "report_step").
        """
        ...

    @property
    @abstractmethod
    def response_type(self) -> str:
        """
        Type alias for the implemented response. For example 'gen_data', 'summary'.
        New types must not conflict with existing types.
        """
        ...

    @property
    @abstractmethod
    def data_layout(self) -> ResponseDataInitialLayout:
        """
        hello
        """
        ...

    @abstractmethod
    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset: ...

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self, dict_factory=CustomDict)
        data["_ert_kind"] = self.__class__.__name__
        return data
