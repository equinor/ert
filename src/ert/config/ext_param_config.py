from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
import polars as pl
import xarray as xr

from ert.substitutions import substitute_runpath_name

from .parameter_config import ParameterCardinality, ParameterConfig, ParameterMetadata

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble

    Number = int | float
    DataType = Mapping[str, Number | Mapping[str, Number]]
    MutableDataType = MutableMapping[str, Number | MutableMapping[str, Number]]


class ExtParamConfig(ParameterConfig):
    """Create an ExtParamConfig for @key with the given @input_keys

    @input_keys can be either a list of keys as strings or a dict with
    keys as strings and a list of suffixes for each key.
    If a list of strings is given, the order is preserved.
    """

    type: Literal["everest_parameters"] = "everest_parameters"
    input_keys: list[str] = field(default_factory=list)
    forward_init: bool = False
    output_file: str = ""
    forward_init_file: str = ""
    update: bool = False
    group: str = ""

    def read_from_runpath(
        self, run_path: Path, real_nr: int, iteration: int
    ) -> xr.Dataset:
        raise NotImplementedError

    def write_to_runpath(
        self, run_path: Path, real_nr: int, ensemble: Ensemble
    ) -> None:
        file_path = run_path / substitute_runpath_name(
            self.output_file, real_nr, ensemble.iteration
        )
        Path.mkdir(file_path.parent, exist_ok=True, parents=True)

        data: MutableDataType = {}
        if file_path.exists():
            data = json.loads(file_path.read_text())
        for da in ensemble.load_parameters(self.name, real_nr).drop("realization"):
            assert isinstance(da, pl.Series)
            name = da.name

            name = name.split(f"{self.group}.")[1]
            da_values = da.to_list()
            try:
                outer, inner = name.split(".")

                if outer not in data:
                    data[outer] = {}
                data[outer][inner] = float(da_values[0])  # type: ignore
            except ValueError:
                data[name] = float(da_values[0])

        file_path.write_text(json.dumps(data), encoding="utf-8")

    def create_storage_datasets(
        self,
        from_data: npt.NDArray[np.float64],
        iens_active_index: npt.NDArray[np.int_],
    ) -> Iterator[tuple[None, pl.DataFrame]]:
        yield (
            None,
            pl.DataFrame(
                {
                    "realization": iens_active_index,
                    self.name: pl.Series(from_data.flatten()),
                }
            ),
        )

    def load_parameters(
        self, ensemble: Ensemble, realizations: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def load_parameter_graph(self) -> nx.Graph[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        return 1

    @property
    def cardinality(self) -> ParameterCardinality:
        return ParameterCardinality.multiple_configs_per_ensemble_dataset

    @property
    def group_name(self) -> str:
        return self.group

    @property
    def parameter_keys(self) -> list[str]:
        return [self.name]

    @property
    def metadata(self) -> list[ParameterMetadata]:
        return []
