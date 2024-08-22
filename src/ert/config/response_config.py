import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal

import xarray as xr

from ert.config.parameter_config import CustomDict


@dataclasses.dataclass
class ResponseConfig(ABC):
    name: str
    input_file: str = ""
    keys: List[str] = dataclasses.field(default_factory=list)

    # Note: This is necessary to handle arbitrary
    # inputs for responses, like report steps for gen data.
    # This is meant to only be accessed from within the
    # response config itself when it is reading files from runpath
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @abstractmethod
    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset: ...

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self, dict_factory=CustomDict)
        data["_ert_kind"] = self.__class__.__name__
        return data

    @staticmethod
    def serialize_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {**kwargs}

    @staticmethod
    def deserialize_kwargs(kwargs_serialized: Dict[str, Any]) -> Dict[str, Any]:
        return {**kwargs_serialized}

    @property
    @abstractmethod
    def cardinality(self) -> Literal["one_per_key", "one_per_realization"]:
        """Specifies how many files are expected from this config"""
        ...

    @property
    @abstractmethod
    def response_type(self) -> str:
        """Label to identify what kind of response it is.
        Must not overlap with that of other response configs."""
        ...
