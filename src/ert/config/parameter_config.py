from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
import polars as pl
import xarray as xr
from pydantic import BaseModel

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble


class CustomDict(dict):  # type: ignore  # noqa: FURB189
    """Used for converting types that can not be serialized
    directly to json
    """

    def __init__(self, data: list[tuple[Any, Any]]) -> None:
        for i, (key, value) in enumerate(data):
            if isinstance(value, Path):
                data[i] = (key, str(value))
            if isinstance(value, set):
                data[i] = (key, list(value))
        super().__init__(data)


class ParameterMetadata(BaseModel):
    key: str
    transformation: str | None
    dimensionality: Literal[1, 2, 3] = 1
    userdata: dict[str, Any]


class ParameterConfig(BaseModel):
    type: str
    name: str
    forward_init: bool
    update: bool

    def sample_or_load(
        self,
        real_nr: int,
        random_seed: int,
    ) -> xr.Dataset | pl.DataFrame:
        return self.read_from_runpath(Path(), real_nr, 0)

    @property
    @abstractmethod
    def parameter_keys(self) -> list[str]:
        """
        Returns a list of parameter keys within this parameter group
        """

    @property
    @abstractmethod
    def metadata(self) -> list[ParameterMetadata]:
        """
        Returns metadata describing this parameter

        """

    @abstractmethod
    def __len__(self) -> int:
        """Number of parameters"""

    @abstractmethod
    def read_from_runpath(
        self,
        run_path: Path,
        real_nr: int,
        iteration: int,
    ) -> xr.Dataset:
        """
        This function is responsible for converting the parameter
        from the forward model to the internal ert format
        """

    @abstractmethod
    def write_to_runpath(
        self, run_path: Path, real_nr: int, ensemble: Ensemble
    ) -> dict[str, dict[str, float | str]] | None:
        """
        This function is responsible for converting the parameter
        from the internal ert format to the format the forward model
        expects
        """

    @abstractmethod
    def save_parameters(
        self,
        ensemble: Ensemble,
        realization: int,
        data: npt.NDArray[np.float64],
    ) -> None:
        """
        Save the parameter in internal storage for the given ensemble
        """

    def copy_parameters(
        self,
        source_ensemble: Ensemble,
        target_ensemble: Ensemble,
        realizations: npt.NDArray[np.int_],
    ) -> None:
        """
        Copy parameters from one ensemble to another.
        If realizations is None, copy all realizations.
        If realizations is given, copy only those realizations.
        """
        for realization in realizations:
            # Converts to standard python scalar due to mypy
            realization_int = int(realization)
            ds = source_ensemble.load_parameters(self.name, realization_int)
            target_ensemble.save_parameters(self.name, realization_int, ds)

    @abstractmethod
    def load_parameters(
        self, ensemble: Ensemble, realizations: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float64]:
        """
        Load the parameter from internal storage for the given ensemble.
        Must return array of shape (number of parameters, number of realizations).
        """

    @abstractmethod
    def load_parameter_graph(self) -> nx.Graph[int]:
        """
        Load the graph encoding Markov properties on the parameter `group`
        Often a neighbourhood graph.
        """

    def save_experiment_data(
        self,
        experiment_path: Path,
    ) -> None:
        pass
