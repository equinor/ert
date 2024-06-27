from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, MutableMapping, Tuple, Union

import numpy as np
import xarray as xr

from .parameter_config import ParameterConfig

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble

    Number = Union[int, float]
    DataType = Mapping[str, Union[Number, Mapping[str, Number]]]
    MutableDataType = MutableMapping[str, Union[Number, MutableMapping[str, Number]]]


@dataclass
class ExtParamConfig(ParameterConfig):
    """Create an ExtParamConfig for @key with the given @input_keys

    @input_keys can be either a list of keys as strings or a dict with
    keys as strings and a list of suffixes for each key.
    If a list of strings is given, the order is preserved.
    """

    input_keys: Union[List[str], Dict[str, List[Tuple[str, str]]]] = field(
        default_factory=list
    )
    forward_init: bool = False
    output_file: str = ""
    forward_init_file: str = ""
    update: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.input_keys, dict):
            for k, suffixes in self.input_keys.items():
                if not isinstance(suffixes, list):
                    raise TypeError(
                        f"Invalid type {type(suffixes)} for suffix: {suffixes}"
                    )

                if len(suffixes) == 0:
                    raise ValueError(
                        f"No suffixes for key '{self.name}/{k}' - suffixes: {suffixes}"
                    )
                if len(suffixes) != len(set(suffixes)):
                    raise ValueError(
                        f"Duplicate suffixes for key '{self.name}/{k}' - "
                        f"suffixes: {suffixes}"
                    )
                if any(len(s) == 0 for s in suffixes):
                    raise ValueError(
                        f"Empty suffix encountered for key '{self.name}/{k}' "
                        f"- suffixes: {suffixes}"
                    )
        else:
            if isinstance(self.input_keys, tuple):
                self.input_keys = list(self.input_keys)
            if len(self.input_keys) != len(set(self.input_keys)):
                raise ValueError(
                    f"Duplicate keys for key '{self.name}' - keys: {self.input_keys}"
                )

    def read_from_runpath(self, run_path: Path, real_nr: int) -> xr.Dataset:
        raise NotImplementedError()

    def write_to_runpath(
        self, run_path: Path, real_nr: int, ensemble: Ensemble
    ) -> None:
        file_path = run_path / self.output_file
        Path.mkdir(file_path.parent, exist_ok=True, parents=True)

        data: MutableDataType = {}
        for da in ensemble.load_parameters(self.name, real_nr)["values"]:
            assert isinstance(da, xr.DataArray)
            name = str(da.names.values)
            try:
                outer, inner = name.split("\0")

                if outer not in data:
                    data[outer] = {}
                data[outer][inner] = float(da)  # type: ignore
            except ValueError:
                data[name] = float(da)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def save_parameters(
        self,
        ensemble: Ensemble,
        group: str,
        realization: int,
        data: npt.NDArray[np.float64],
    ) -> None:
        raise NotImplementedError()

    def load_parameters(
        self, ensemble: Ensemble, group: str, realizations: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError()

    @staticmethod
    def to_dataset(data: DataType) -> xr.Dataset:
        """Flattens data to fit inside a dataset"""
        names: List[str] = []
        values: List[float] = []
        for outer_key, outer_val in data.items():
            if isinstance(outer_val, (int, float)):
                names.append(outer_key)
                values.append(float(outer_val))
                continue
            for inner_key, inner_val in outer_val.items():
                names.append(f"{outer_key}\0{inner_key}")
                values.append(float(inner_val))

        return xr.Dataset(
            {
                "values": ("names", np.array(values, dtype=np.float64)),
                "names": names,
            }
        )

    def __len__(self) -> int:
        return len(self.input_keys)

    def __contains__(self, key: Union[Tuple[str, str], str]) -> bool:
        """Check if the @key is present in the configuration
        @key can be a single string or a tuple (key, suffix)
        """
        if isinstance(self.input_keys, dict) and isinstance(key, tuple):
            key, suffix = key
            return (
                key in self.input_keys and suffix in self.input_keys[key]  # type: ignore[comparison-overlap]
            )
        else:
            return key in self.input_keys

    def __repr__(self) -> str:
        return f"ExtParamConfig(keys={self.input_keys})"

    def __getitem__(self, index: str) -> List[Tuple[str, str]]:
        """Retrieve an item from the configuration

        If @index is a string, assumes its a key and retrieves the suffixes
        for that key
        An IndexError is raised if the item is not found
        """
        if not isinstance(index, str):
            raise IndexError(
                f"Unexpected index of type {type(index)} for Keylist: {self.input_keys}"
            )
        if isinstance(self.input_keys, dict):
            if index in self.input_keys:
                return self.input_keys[index]
            else:
                raise IndexError(
                    f"Requested index not found: {index},"
                    f"Keylist: {list(self.input_keys.keys())}"
                )
        elif isinstance(self.input_keys, list):
            if index in self.input_keys:
                return []
            raise IndexError(f"Requested index not found: {index}")
        else:
            raise IndexError(
                f"Unexpected index of type {type(index)} for Keylist: {self.input_keys}"
            )
