from pathlib import Path
from typing import Self

import networkx as nx
import numpy as np
import xarray as xr
from pydantic import BaseModel, model_validator

# Assuming these are imported properly
from .parameter_config import ParameterConfig, ParameterMetadata


class ParameterGroupConfig(BaseModel):
    parameters: list[ParameterConfig]
    parameter_type: str | None = None

    @model_validator(mode="after")
    def enforce_same_type(self) -> Self:
        if not self.parameters:
            raise ValueError("Parameter list cannot be empty")

        first_type = type(self.parameters[0])
        if not all(isinstance(p, first_type) for p in self.parameters):
            raise ValueError(
                f"All parameters must be of the same type. "
                f"Got types: {[type(p).__name__ for p in self.parameters]}"
            )

        param_type_str = self.parameters[0].type
        self.parameter_type = param_type_str

        if param_type_str != "gen_kw" and len(self.parameters) != 1:
            raise ValueError("Only 'gen_kw' can appear in groups of length > 1.")
        return self

    @property
    def parameter_keys(self) -> list[str]:
        return [key for p in self.parameters for key in p.parameter_keys]

    @property
    def metadata(self) -> list[ParameterMetadata]:
        return [m for p in self.parameters for m in p.metadata]

    def __len__(self) -> int:
        return sum(len(p) for p in self.parameters)

    def read_from_runpath(
        self, run_path: Path, real_nr: int, iteration: int
    ) -> xr.Dataset:
        # Combine datasets from all parameter configs
        datasets = [
            p.read_from_runpath(run_path, real_nr, iteration) for p in self.parameters
        ]
        return xr.merge(datasets)

    def write_to_runpath(
        self, run_path: Path, real_nr: int, ensemble
    ) -> dict[str, dict[str, float | str]] | None:
        result = {}
        for p in self.parameters:
            p_result = p.write_to_runpath(run_path, real_nr, ensemble)
            if p_result is not None:
                result.update(p_result)
        return result or None

    # def create_storage_datasets(
    #     self,
    #     from_data: npt.NDArray[np.float64],
    #     iens_active_index: npt.NDArray[np.int_],
    # ) -> Iterator[tuple[int | None, pl.DataFrame | xr.Dataset]]:
    #     params_datasets: list[xr.Dataset | pl.DataFrame] = []
    #     for p in self.parameters:
    #         p.create_storage_datasets
    #         yield from p.create_storage_datasets(
    #             from_data[offset : offset + param_len, :], iens_active_index
    #         )
    #         offset += param_len

    # def load_parameters(self, ensemble, realizations) -> xr.Dataset | pl.DataFrame:
    #     # Vertically stack all loaded parameters
    #     return np.vstack(
    #         [p.load_parameters(ensemble, realizations) for p in self.parameters]
    #     )

    def load_parameter_graph(self) -> nx.Graph:
        # Combine all graphs
        graphs = [p.load_parameter_graph() for p in self.parameters]
        combined = nx.disjoint_union_all(graphs)
        return combined

    def save_experiment_data(self, experiment_path: Path) -> None:
        for p in self.parameters:
            p.save_experiment_data(experiment_path)

    @staticmethod
    def sample_value(
        parameter_group_name: str,
        keys: list[str],
        global_seed: str,
        realization: int,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Call sample_value on each sub-parameter individually, not the group."
        )
