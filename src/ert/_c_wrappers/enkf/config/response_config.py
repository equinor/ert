import dataclasses
from abc import ABC, abstractmethod

import xarray as xr


@dataclasses.dataclass
class ResponseConfig(ABC):
    name: str

    @abstractmethod
    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset:
        ...
