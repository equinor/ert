import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal

import xarray as xr

from ert.config.parameter_config import CustomDict


@dataclasses.dataclass
class ResponseConfig(ABC):
    name: str

    @property
    @abstractmethod
    def dataset_cardinality(
        self,
    ) -> Literal["one_file_per_key", "one_file_per_realization"]: ...

    @property
    @abstractmethod
    def response_type(
        self,
    ) -> str: ...

    @abstractmethod
    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset: ...

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self, dict_factory=CustomDict)
        data["_ert_kind"] = self.__class__.__name__
        return data
