import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict

import xarray as xr

from ert.config.parameter_config import CustomDict


class InvalidResponseFile(Exception):
    """
    Raised when an input file of the ResponseConfig has
    the incorrect format.
    """


@dataclasses.dataclass
class ResponseConfig(ABC):
    name: str

    @abstractmethod
    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset:
        """Reads the data for the response from run_path.

        Raises:
            FileNotFoundError: when one of the input_files for the
                response is missing.
            InvalidResponseFile: when one of the input_files is
                invalid
        """

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self, dict_factory=CustomDict)
        data["_ert_kind"] = self.__class__.__name__
        return data
