from __future__ import annotations

import json
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, MutableMapping, Optional, Tuple, Union
from uuid import UUID

import numpy as np
import xarray as xr
from pydantic import BaseModel

from ert.realization_state import RealizationState
from ert.storage.local_realization import LocalRealization

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage.local_experiment import (
        LocalExperimentAccessor,
        LocalExperimentReader,
    )
    from ert.storage.local_storage import LocalStorageAccessor, LocalStorageReader

logger = logging.getLogger(__name__)


class _Index(BaseModel):
    id: UUID
    experiment_id: UUID
    ensemble_size: int
    iteration: int
    name: str
    prior_ensemble_id: Optional[UUID]
    started_at: datetime


class LocalEnsembleReader:
    def __init__(
        self,
        storage: LocalStorageReader,
        path: Path,
    ):
        self._storage: Union[LocalStorageReader, LocalStorageAccessor] = storage
        self._path = path
        self._index = _Index.parse_file(path / "index.json")
        self._experiment_path = self._path / "experiment"

        self._state_map = self._load_state_map()
        self._realizations: MutableMapping[
            Tuple[int, Literal["r", "w"]], LocalRealization
        ] = {}

    @property
    def mount_point(self) -> Path:
        return self._path

    @property
    def name(self) -> str:
        return self._index.name

    @property
    def id(self) -> UUID:
        return self._index.id

    @property
    def experiment_id(self) -> UUID:
        return self._index.experiment_id

    @property
    def ensemble_size(self) -> int:
        return self._index.ensemble_size

    @property
    def started_at(self) -> datetime:
        return self._index.started_at

    @property
    def iteration(self) -> int:
        return self._index.iteration

    @property
    def state_map(self) -> List[RealizationState]:
        return self._state_map

    @property
    def experiment(self) -> Union[LocalExperimentReader, LocalExperimentAccessor]:
        return self._storage.get_experiment(self.experiment_id)

    def close(self) -> None:
        self.sync()

    def sync(self) -> None:
        pass

    def get_realization_mask_from_state(
        self, states: List[RealizationState]
    ) -> npt.NDArray[np.bool_]:
        return np.array([s in states for s in self._state_map], dtype=bool)

    def get_realization(
        self, index: int, mode: Literal["r", "w"] = "r"
    ) -> LocalRealization:
        if (real := self._realizations.get((index, mode))) is not None:
            return real
        real = self._realizations[(index, mode)] = LocalRealization(self, index, mode)
        return real

    def _load_state_map(self) -> List[RealizationState]:
        state_map_file = self._experiment_path / "state_map.json"
        if state_map_file.exists():
            with open(state_map_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [RealizationState(v) for v in data["state_map"]]
        else:
            return [RealizationState.UNDEFINED for _ in range(self.ensemble_size)]

    @property
    def is_initalized(self) -> bool:
        return RealizationState.INITIALIZED in self.state_map or self.has_data

    @property
    def has_data(self) -> bool:
        return RealizationState.HAS_DATA in self.state_map

    def realizations_initialized(self, realizations: List[int]) -> bool:
        initialized_realizations = set(
            self.realization_list(RealizationState.INITIALIZED)
        )
        return all(real in initialized_realizations for real in realizations)

    def get_summary_keyset(self) -> List[str]:
        realization_folders = list(self.mount_point.glob("realization-*"))
        if not realization_folders:
            return []
        summary_path = realization_folders[0] / "summary.nc"
        if not summary_path.exists():
            return []
        realization_nr = int(str(realization_folders[0])[-1])
        response = self.load_response("summary", (realization_nr,))
        keys = sorted(response["name"].values)
        return keys

    def realization_list(self, state: RealizationState) -> List[int]:
        """
        Will return list of realizations with state == the specified state.
        """
        return [i for i, s in enumerate(self._state_map) if s == state]

    def _load_dataset(
        self,
        group: str,
        realizations: Union[int, npt.NDArray[np.int_], None],
    ) -> xr.Dataset:
        if isinstance(realizations, int):
            return (
                self.get_realization(realizations)
                .load_dataset(group)
                .isel(realizations=0, drop=True)
            )

        if realizations is None:
            datasets = [
                xr.open_dataset(p, engine="scipy")
                for p in sorted(self.mount_point.glob(f"realization-*/{group}.nc"))
            ]
        else:
            datasets = [
                self.get_realization(i).load_dataset(group) for i in realizations
            ]
        return xr.combine_nested(datasets, "realizations")

    def has_parameter_group(self, group: str) -> bool:
        param_group_file = self.mount_point / f"realization-0/{group}.nc"
        return param_group_file.exists()

    def load_parameters(
        self,
        group: str,
        realizations: Union[int, npt.NDArray[np.int_], None] = None,
        *,
        var: str = "values",
    ) -> xr.DataArray:
        return self._load_dataset(group, realizations)[var]

    @lru_cache  # noqa: B019
    def load_response(self, key: str, realizations: Tuple[int, ...]) -> xr.Dataset:
        loaded = []
        for realization in realizations:
            input_path = self.mount_point / f"realization-{realization}" / f"{key}.nc"
            if not input_path.exists():
                raise KeyError(f"No response for key {key}, realization: {realization}")
            ds = xr.open_dataset(input_path, engine="scipy")
            loaded.append(ds)
        response = xr.combine_nested(loaded, concat_dim="realization")
        assert isinstance(response, xr.Dataset)
        return response


class LocalEnsembleAccessor(LocalEnsembleReader):
    def __init__(
        self,
        storage: LocalStorageAccessor,
        path: Path,
    ):
        super().__init__(storage, path)
        self._storage: LocalStorageAccessor = storage

    @classmethod
    def create(
        cls,
        storage: LocalStorageAccessor,
        path: Path,
        uuid: UUID,
        *,
        ensemble_size: int,
        experiment_id: UUID,
        iteration: int = 0,
        name: str,
        prior_ensemble_id: Optional[UUID],
    ) -> LocalEnsembleAccessor:
        (path / "experiment").mkdir(parents=True, exist_ok=False)

        index = _Index(
            id=uuid,
            ensemble_size=ensemble_size,
            experiment_id=experiment_id,
            iteration=iteration,
            name=name,
            prior_ensemble_id=prior_ensemble_id,
            started_at=datetime.now(),
        )

        with open(path / "index.json", mode="w", encoding="utf-8") as f:
            print(index.json(), file=f)

        return cls(storage, path)

    def _save_state_map(self) -> None:
        state_map_file = self._experiment_path / "state_map.json"
        with open(state_map_file, "w", encoding="utf-8") as f:
            data = {"state_map": [v.value for v in self._state_map]}
            f.write(json.dumps(data))

    def update_realization_state(
        self,
        realization: int,
        old_states: List[RealizationState],
        new_state: RealizationState,
    ) -> None:
        if self._state_map[realization] in old_states:
            self._state_map[realization] = new_state

    def sync(self) -> None:
        self._save_state_map()

    def save_parameters(
        self,
        group: str,
        realization: int,
        dataset: xr.Dataset,
    ) -> None:
        """Saves the provided dataset under a parameter group and realization index

        Args:
            group: Name of the parameter group under which the dataset is to be saved

            realization: Which realization index this group belongs to

            dataset: Dataset to save. It must contain a variable named
                    'values' which will be used when flattening out the
                    parameters into a 1d-vector.
        """
        self.get_realization(realization, "w").save_parameters(group, dataset)
        self.update_realization_state(
            realization,
            [
                RealizationState.UNDEFINED,
                RealizationState.LOAD_FAILURE,
            ],
            RealizationState.INITIALIZED,
        )

    def save_response(self, group: str, data: xr.Dataset, realization: int) -> None:
        self.get_realization(realization, "w").save_response(group, data)
