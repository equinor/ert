from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Callable, Iterator
from enum import StrEnum, auto
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
import polars as pl
import xarray as xr
from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble

logger = logging.getLogger(__name__)

EXT4_MAX_BYTE_LENGTH = 255


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

    @model_validator(mode="after")
    def log_parameters_on_instantiation(self) -> ParameterConfig:
        specified_parameters = self.model_fields_set
        defaulted_parameters = set(self.model_fields.keys()) - specified_parameters
        msg = (
            f"Attributes for {type(self).__name__} with input values:\n"
            f"{specified_parameters}\n"
            f"Attributes for {type(self).__name__} with defaulted values:\n"
            f"{defaulted_parameters}"
        )
        logger.info(msg)
        return self

    @model_validator(mode="after")
    def validate_name_length(self) -> ParameterConfig:
        byte_representation_of_name = self.name.encode("utf8")
        bytes_in_name = len(byte_representation_of_name)
        if bytes_in_name > EXT4_MAX_BYTE_LENGTH:
            raise InvalidParameterFile(
                f"The byte size '{bytes_in_name}' of parameter name '{self.name}' "
                f"exceeds maximum size '{EXT4_MAX_BYTE_LENGTH}'.\n"
                f"Consider shortening this parameter name."
            )
        return self

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
            target_ensemble.save_parameters(ds, self.name, realization_int)

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

    def sample_value(
        self,
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
        key_hash = sha256(
            global_seed.encode("utf-8") + f"{self.group_name}:{self.name}".encode()
        )
        seed = np.frombuffer(key_hash.digest(), dtype="uint32")
        rng = np.random.default_rng(seed)

        # Advance the RNG state to the realization point
        rng.standard_normal(realization)

        # Generate a single sample
        value = rng.standard_normal(1)
        return np.array([value[0]])
