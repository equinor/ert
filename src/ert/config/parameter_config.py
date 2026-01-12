from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterator
from enum import StrEnum, auto
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import polars as pl
import xarray as xr
from pydantic import BaseModel

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble


class InvalidParameterFile(Exception):
    """
    Raised when a parameter file does not fulfill its
    format requirements.
    """


class ParameterCardinality(StrEnum):
    """
    multiple_configs_per_ensemble_dataset: multiple config instances per group, one
    dataset per ensemble

    one_config_per_realization_dataset: one config instance per group, one
    dataset per realization
    """

    multiple_configs_per_ensemble_dataset = auto()
    one_config_per_realization_dataset = auto()


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
    ) -> Iterator[tuple[int | None, pl.DataFrame | xr.Dataset, str | None]]:
        """
        Iterates over realization. It creates an xarray Dataset
        or polars DataFrame from the numpy data
        """

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

    @property
    def cardinality(self) -> ParameterCardinality:
        return ParameterCardinality.one_config_per_realization_dataset

    def save_experiment_data(
        self,
        experiment_path: Path,
    ) -> None:
        pass

    @property
    def group_name(self) -> str:
        return self.name

    def transform_data(self) -> Callable[[float], float]:
        return lambda x: x

    def sample_values(
        self, global_seed: str, active_realizations: list[int], num_realizations: int
    ) -> npt.NDArray[np.double]:
        """
        Generate reproducible standard-normal samples for active realizations.

        For this parameter (identified by self.group_name and self.name), a random
        sampling of size `num_realizations` is constructed using an RNG
        seeded from `global_seed` and the parameter name. The entries at the
        indices specified by `active_realizations` are then returned.

        Parameters:
        - global_seed (str): A global seed string used for RNG seed generation to
        ensure reproducibility across runs.
        - active_realizations (list[int]): indices of the realizations
        to select from the sampling vector; each must satisfy 0 <= i < num_realizations.
        - num_realizations (int): Total number of realizations. Assures stable sampling
        for a given global_seed regardless of currently active realizations.

        Returns:
        - npt.NDArray[np.double]: Array of shape (len(active_realizations),
        containing sample values, one for each `active_realization`.
        """
        key_hash = sha256(
            global_seed.encode("utf-8") + f"{self.group_name}:{self.name}".encode()
        )
        seed = np.frombuffer(key_hash.digest(), dtype="uint32")
        rng = np.random.default_rng(seed)

        # Generate samples for all active realizations
        all_values = rng.standard_normal(num_realizations)
        return all_values[active_realizations]
