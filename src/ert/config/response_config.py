import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict

import xarray as xr

from ert.config.parameter_config import CustomDict


@dataclasses.dataclass
class ResponseConfig(ABC):
    name: str

    @abstractmethod
    def read_from_file(self, file_path: str) -> xr.Dataset: ...

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self, dict_factory=CustomDict)
        data["_ert_kind"] = self.__class__.__name__
        return data
