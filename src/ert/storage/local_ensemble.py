from __future__ import annotations

import json
import logging
from datetime import datetime
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
from uuid import UUID

import numpy as np
import xarray as xr
from pydantic import BaseModel

from ert._c_wrappers.enkf.config.field_config import field_transform
from ert._c_wrappers.enkf.enums import RealizationStateEnum
from ert._c_wrappers.enkf.time_map import TimeMap
from ert.callbacks import forward_model_ok
from ert.load_status import LoadResult, LoadStatus
from ert.storage.field_utils import field_utils

if TYPE_CHECKING:
    import numpy.typing as npt
    from ecl.summary import EclSum

    from ert._c_wrappers.enkf.ensemble_config import EnsembleConfig
    from ert._c_wrappers.enkf.run_arg import RunArg
    from ert.storage.local_experiment import (
        LocalExperimentAccessor,
        LocalExperimentReader,
    )
    from ert.storage.local_storage import LocalStorageAccessor, LocalStorageReader

logger = logging.getLogger(__name__)


def _load_realization(
    sim_fs: LocalEnsembleAccessor,
    realisation: int,
    ensemble_config: EnsembleConfig,
    run_args: List[RunArg],
) -> Tuple[LoadResult, int]:
    sim_fs.update_realization_state(
        realisation,
        [RealizationStateEnum.STATE_UNDEFINED],
        RealizationStateEnum.STATE_INITIALIZED,
    )
    result = forward_model_ok(run_args[realisation], ensemble_config)
    sim_fs.state_map[realisation] = (
        RealizationStateEnum.STATE_HAS_DATA
        if result.status == LoadStatus.LOAD_SUCCESSFUL
        else RealizationStateEnum.STATE_LOAD_FAILURE
    )
    return result, realisation


def _field_truncate(data: npt.ArrayLike, min_: float, max_: float) -> Any:
    if min_ is not None and max_ is not None:
        vfunc = np.vectorize(lambda x: max(min(x, max_), min_))
        return vfunc(data)
    elif min_ is not None:
        vfunc = np.vectorize(lambda x: max(x, min_))
        return vfunc(data)
    elif max_ is not None:
        vfunc = np.vectorize(lambda x: min(x, max_))
        return vfunc(data)
    return data


class _Index(BaseModel):
    id: UUID
    experiment_id: UUID
    ensemble_size: int
    iteration: int
    name: str
    prior_ensemble_id: Optional[UUID]
    started_at: datetime


# pylint: disable=R0904
class LocalEnsembleReader:
    def __init__(
        self,
        storage: LocalStorageReader,
        path: Path,
        refcase: Optional[EclSum],
    ):
        self._storage: Union[LocalStorageReader, LocalStorageAccessor] = storage
        self._path = path
        self._index = _Index.parse_file(path / "index.json")
        self._experiment_path = self._path / "experiment"

        self._time_map = TimeMap()
        self._time_map.read(self._path / "time_map")
        if refcase:
            self._time_map.attach_refcase(refcase)
        self._state_map = self._load_state_map()

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
    def time_map(self) -> TimeMap:
        return self._time_map

    @property
    def state_map(self) -> List[RealizationStateEnum]:
        return self._state_map

    @property
    def experiment(self) -> Union[LocalExperimentReader, LocalExperimentAccessor]:
        return self._storage.get_experiment(self.experiment_id)

    def close(self) -> None:
        self.sync()

    def sync(self) -> None:
        pass

    def get_realization_mask_from_state(
        self, states: List[RealizationStateEnum]
    ) -> List[bool]:
        return [s in states for s in self._state_map]

    def _load_state_map(self) -> List[RealizationStateEnum]:
        state_map_file = self._experiment_path / "state_map.json"
        if state_map_file.exists():
            with open(state_map_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [RealizationStateEnum(v) for v in data["state_map"]]
        else:
            return [
                RealizationStateEnum.STATE_UNDEFINED for _ in range(self.ensemble_size)
            ]

    @property
    def is_initalized(self) -> bool:
        return RealizationStateEnum.STATE_INITIALIZED in self.state_map or self.has_data

    @property
    def has_data(self) -> bool:
        return RealizationStateEnum.STATE_HAS_DATA in self.state_map

    def realizations_initialized(self, realizations: List[int]) -> bool:
        initialized_realizations = set(
            self.realization_list(RealizationStateEnum.STATE_INITIALIZED)
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

    def realization_list(self, state: RealizationStateEnum) -> List[int]:
        """
        Will return list of realizations with state == the specified state.
        """
        return [i for i, s in enumerate(self._state_map) if s == state]

    def _load_single_dataset(
        self,
        group: str,
        realization: int,
    ) -> xr.Dataset:
        try:
            return xr.open_dataset(
                self.mount_point / f"realization-{realization}" / group, engine="scipy"
            )
        except FileNotFoundError:
            raise KeyError(
                f"No dataset '{group}' in storage for realization {realization}"
            )

    def _load_dataset(
        self,
        group: str,
        realizations: Union[int, Sequence[int], None],
    ) -> xr.Dataset:
        if isinstance(realizations, int):
            return self._load_single_dataset(group, realizations).isel(
                realizations=0, drop=True
            )

        if realizations is None:
            datasets = [
                xr.open_dataset(p, engine="scipy")
                for p in sorted(self.mount_point.glob(f"realization-*/{group}"))
            ]
        else:
            datasets = [self._load_single_dataset(group, i) for i in realizations]
        return xr.combine_nested(datasets, "realizations")

    def load_parameters(
        self,
        group: str,
        realizations: Union[int, Sequence[int], None] = None,
        *,
        var: str = "values",
    ) -> xr.DataArray:
        return self._load_dataset(group, realizations)[var]

    @lru_cache
    def load_response(self, key: str, realizations: Tuple[int, ...]) -> xr.Dataset:
        loaded = []
        for realization in realizations:
            input_path = self.mount_point / f"realization-{realization}" / f"{key}.nc"
            if not input_path.exists():
                raise KeyError(f"No response for key {key}, realization: {realization}")
            ds = xr.open_dataset(input_path, engine="scipy")
            loaded.append(ds)
        response = xr.combine_by_coords(loaded)
        assert isinstance(response, xr.Dataset)
        return response

    def field_has_data(self, key: str, realization: int) -> bool:
        path = self.mount_point / f"realization-{realization}/{key}"
        return path.exists()

    def export_field(
        self,
        key: str,
        realization: int,
        output_path: Path,
        fformat: Optional[str] = None,
    ) -> None:
        info = self.experiment.parameter_info[key]
        if fformat is None:
            fformat = info["file_format"]

        data_path = self.mount_point / f"realization-{realization}"

        if not data_path.exists():
            raise KeyError(
                f"Unable to load FIELD for key: {key}, realization: {realization} "
            )
        da = xr.open_dataarray(data_path / f"{key}")
        # Squeeze to get rid of realization-dimension
        data: npt.NDArray[np.double] = da.values.squeeze(axis=0)
        data = field_transform(data, transform_name=info["output_transformation"])
        data = _field_truncate(
            data,
            info["truncation_min"],
            info["truncation_max"],
        )

        field_utils.save_field(
            data,
            key,
            self.experiment.grid_path,
            field_utils.Shape(info["nx"], info["ny"], info["nz"]),
            output_path,
            fformat,
        )


class LocalEnsembleAccessor(LocalEnsembleReader):
    def __init__(
        self,
        storage: LocalStorageAccessor,
        path: Path,
        refcase: Optional[EclSum],
    ):
        super().__init__(storage, path, refcase)
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
        refcase: Optional[EclSum],
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

        return cls(storage, path, refcase=refcase)

    def _save_state_map(self) -> None:
        state_map_file = self._experiment_path / "state_map.json"
        with open(state_map_file, "w", encoding="utf-8") as f:
            data = {"state_map": [v.value for v in self._state_map]}
            f.write(json.dumps(data))

    def update_realization_state(
        self,
        realization: int,
        old_states: List[RealizationStateEnum],
        new_state: RealizationStateEnum,
    ) -> None:
        if self._state_map[realization] in old_states:
            self._state_map[realization] = new_state

    def sync(self) -> None:
        self._save_state_map()
        self.time_map.write(str(self._experiment_path / "time_map"))

    def load_from_run_path(
        self,
        ensemble_size: int,
        ensemble_config: EnsembleConfig,
        run_args: List[RunArg],
        active_realizations: List[bool],
    ) -> int:
        """Returns the number of loaded realizations"""
        pool = ThreadPool(processes=8)

        async_result = [
            pool.apply_async(
                _load_realization,
                (self, iens, ensemble_config, run_args),
            )
            for iens in range(ensemble_size)
            if active_realizations[iens]
        ]

        loaded = 0
        for t in async_result:
            ((status, message), iens) = t.get()

            if status == LoadStatus.LOAD_SUCCESSFUL:
                loaded += 1
                self.state_map[iens] = RealizationStateEnum.STATE_HAS_DATA
            else:
                logger.error(f"Realization: {iens}, load failure: {message}")

        return loaded

    def save_parameters(
        self, group: str, realization: int, dataset: Union[xr.DataArray, xr.Dataset]
    ) -> None:
        """Create a parameter group

        Args:
            name: Name of the parameter group

            realization_index: Which realization index this group belongs to

            dataset: Dataset to save. One of the variables must be named
                'value'. This will be used when flattening out the parameters
                into a 1d-vector.

        """
        if isinstance(dataset, xr.DataArray):
            dataset = dataset.to_dataset()

        if "values" not in dataset.variables:
            raise ValueError(
                f"Dataset for parameter group '{group}' "
                f"must contain a 'values' variable"
            )

        path = self.mount_point / f"realization-{realization}" / group
        path.parent.mkdir(exist_ok=True)
        dataset.expand_dims(realizations=[realization]).to_netcdf(path, engine="scipy")
        self.update_realization_state(
            realization,
            [
                RealizationStateEnum.STATE_UNDEFINED,
                RealizationStateEnum.STATE_LOAD_FAILURE,
            ],
            RealizationStateEnum.STATE_INITIALIZED,
        )

    def save_response(self, group: str, data: xr.Dataset, realization: int) -> None:
        data = data.expand_dims({"realization": [realization]})
        output_path = self.mount_point / f"realization-{realization}"
        Path.mkdir(output_path, parents=True, exist_ok=True)

        data.to_netcdf(output_path / f"{group}.nc", engine="scipy")
