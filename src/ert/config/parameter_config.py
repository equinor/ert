from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from hashlib import sha256
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

        Returns:
            Optionally returns a mapping from parameter name to
            parameter value.
        """

    @abstractmethod
    def create_storage_datasets(
        self,
        from_data: npt.NDArray[np.float64],
        iens_active_index: npt.NDArray[np.int_],
    ) -> Iterator[tuple[int | None, pl.DataFrame | xr.Dataset]]:
        """
        Iterates over realization. It creates an xarray Dataset
        or polars DataFrame from the numpy data
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

    @staticmethod
    def sample_value(
        parameter_group_name: str,
        keys: list[str],
        global_seed: str,
        realization: int,
    ) -> npt.NDArray[np.double]:
        """
        Generate a sample value for each key in a parameter group.

        The sampling is reproducible and dependent on a global seed combined
        with the parameter group name and individual key names. The 'realization'
        parameter determines the specific sample point from the distribution for each
        parameter.

        Parameters:
        - parameter_group_name (str): The name of the parameter group, used to ensure
        unique RNG seeds for different groups.
        - keys (list[str]): A list of parameter keys for which the sample values are
        generated.
        - global_seed (str): A global seed string used for RNG seed generation to ensure
        reproducibility across runs.
        - realization (int): An integer used to advance the RNG to a specific point in
        its sequence, effectively selecting the 'realization'-th sample from the
        distribution.

        Returns:
        - npt.NDArray[np.double]: An array of sample values, one for each key in the
        provided list.

        Note:
        The method uses SHA-256 for hash generation and numpy's default random number
        generator for sampling. The RNG state is advanced to the 'realization' point
        before generating a single sample, enhancing efficiency by avoiding the
        generation of large, unused sample sets.
        """
        parameter_values = []
        for key in keys:
            key_hash = sha256(
                global_seed.encode("utf-8") + f"{parameter_group_name}:{key}".encode()
            )
            seed = np.frombuffer(key_hash.digest(), dtype="uint32")
            rng = np.random.default_rng(seed)

            # Advance the RNG state to the realization point
            rng.standard_normal(realization)

            # Generate a single sample
            value = rng.standard_normal(1)
            parameter_values.append(value[0])
        return np.array(parameter_values)
