from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Generator,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    overload,
)
from uuid import UUID

import xarray as xr
import xtgeo

from ert.config import EnsembleConfig, ParameterConfig
from ert.realization_state import RealizationState
from ert.run_arg import RunArg


class StorageReader(Protocol):
    path: Path

    def refresh(self) -> None:
        ...

    def close(self) -> None:
        ...

    def to_accessor(self) -> StorageAccessor:
        ...

    @overload
    def get_experiment(self, uuid: UUID, mode: Literal["r"] = "r") -> ExperimentReader:
        ...

    @overload
    def get_experiment(self, uuid: UUID, mode: Literal["w"]) -> ExperimentAccessor:
        ...

    @overload
    def get_ensemble(self, uuid: UUID, mode: Literal["r"] = "r") -> EnsembleReader:
        ...

    @overload
    def get_ensemble(self, uuid: UUID, mode: Literal["w"]) -> EnsembleAccessor:
        ...

    @overload
    def get_ensemble_by_name(
        self, name: str, mode: Literal["r"] = "r"
    ) -> EnsembleReader:
        ...

    @overload
    def get_ensemble_by_name(self, name: str, mode: Literal["w"]) -> EnsembleAccessor:
        ...

    @property
    def ensembles(self) -> Generator[EnsembleReader, None, None]:
        ...

    @property
    def experiments(self) -> Generator[ExperimentReader, None, None]:
        ...


class StorageAccessor(StorageReader, Protocol):
    def create_experiment(
        self, parameters: Optional[Sequence[ParameterConfig]] = None
    ) -> ExperimentAccessor:
        ...

    def create_ensemble(
        self,
        experiment: Union[ExperimentReader, UUID],
        *,
        ensemble_size: int,
        iteration: int = 0,
        name: Optional[str] = None,
        prior_ensemble: Union[EnsembleReader, UUID, None] = None,
    ) -> EnsembleAccessor:
        ...


class ExperimentReader(Protocol):
    @property
    def id(self) -> UUID:
        ...

    @property
    def parameter_info(self) -> Mapping[str, Any]:
        ...

    @property
    def parameter_configuration(self) -> Mapping[str, ParameterConfig]:
        ...

    @property
    def ensembles(self) -> Generator[EnsembleReader, None, None]:
        ...

    def get_surface(self, name: str) -> xtgeo.RegularSurface:
        ...


class ExperimentAccessor(ExperimentReader, Protocol):
    def create_ensemble(
        self,
        *,
        ensemble_size: int,
        iteration: int = 0,
        name: str,
        prior_ensemble: Optional[EnsembleReader] = None,
    ) -> EnsembleAccessor:
        ...


class EnsembleReader(Protocol):
    @property
    def state_map(self) -> List[RealizationState]:
        ...

    @property
    def mount_point(self) -> Path:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def id(self) -> UUID:
        ...

    @property
    def experiment_id(self) -> UUID:
        ...

    @property
    def ensemble_size(self) -> int:
        ...

    @property
    def started_at(self) -> datetime:
        ...

    @property
    def iteration(self) -> int:
        ...

    @property
    def experiment(self) -> ExperimentReader:
        ...

    def close(self) -> None:
        ...

    def sync(self) -> None:
        ...

    def get_realization_mask_from_state(
        self, states: List[RealizationState]
    ) -> List[bool]:
        ...

    @property
    def is_initalized(self) -> bool:
        ...

    @property
    def has_data(self) -> bool:
        ...

    def realizations_initialized(self, realizations: List[int]) -> bool:
        ...

    def get_summary_keyset(self) -> List[str]:
        ...

    def realization_list(self, state: RealizationState) -> List[int]:
        ...

    def load_parameters(
        self,
        group: str,
        realizations: Union[int, Sequence[int], None] = None,
        *,
        var: str = "values",
    ) -> xr.DataArray:
        ...

    def load_response(self, key: str, realizations: Tuple[int, ...]) -> xr.Dataset:
        ...


class EnsembleAccessor(EnsembleReader, Protocol):
    def update_realization_state(
        self,
        realization: int,
        old_states: Sequence[RealizationState],
        new_state: RealizationState,
    ) -> None:
        pass

    def load_from_run_path(
        self,
        ensemble_size: int,
        ensemble_config: EnsembleConfig,
        run_args: Sequence[RunArg],
        active_realizations: List[bool],
    ) -> int:
        ...

    def save_parameters(
        self, group: str, realization: int, dataset: xr.Dataset
    ) -> None:
        ...

    def save_response(self, group: str, data: xr.Dataset, realization: int) -> None:
        ...
