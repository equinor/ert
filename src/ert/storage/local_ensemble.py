from __future__ import annotations

import json
import logging
import math
import shutil
from datetime import datetime
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union
from uuid import UUID

import numpy as np
import xarray as xr
import xtgeo
from pydantic import BaseModel

from ert._c_wrappers.enkf.config.field_config import field_transform
from ert._c_wrappers.enkf.config.gen_kw_config import PRIOR_FUNCTIONS
from ert._c_wrappers.enkf.enums import RealizationStateEnum
from ert._c_wrappers.enkf.time_map import TimeMap
from ert.callbacks import forward_model_ok
from ert.load_status import LoadResult, LoadStatus
from ert.parsing import ConfigValidationError
from ert.storage.field_utils.field_utils import Shape, read_mask, save_field

if TYPE_CHECKING:
    import numpy.typing as npt
    from ecl.summary import EclSum
    from xtgeo import RegularSurface

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
        return self.has_parameters()

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

    def load_gen_kw_as_dict(
        self, key: str, realization: int
    ) -> Dict[str, Dict[str, float]]:
        data, keys = self.load_gen_kw_realization(key, realization)
        priors = {p["key"]: p for p in self.experiment.gen_kw_info[key]}

        transformed = {
            parameter_key: PRIOR_FUNCTIONS[priors[parameter_key]["function"]](
                value, list(priors[parameter_key]["parameters"].values())
            )
            for parameter_key, value in zip(keys, data)
        }

        result = {key: transformed}

        log10 = {
            parameter_key: math.log(value, 10)
            for parameter_key, value in transformed.items()
            if "LOG" in priors[parameter_key]["function"]
        }
        if log10:
            result.update({f"LOG10_{key}": log10})
        return result

    def load_gen_kw(self, key: str, realizations: List[int]) -> npt.NDArray[np.double]:
        result = []
        for realization in realizations:
            data, _ = self.load_gen_kw_realization(key, realization)
            result.append(data)
        return np.stack(result).T

    def load_ext_param(self, key: str, realization: int) -> Any:
        input_path = self.mount_point / f"realization-{realization}" / f"{key}.json"
        if not input_path.exists():
            raise KeyError(f"No parameter: {key} in storage")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def has_surface(self, key: str, realization: int) -> bool:
        input_path = self.mount_point / f"realization-{realization}"
        return (input_path / f"{key}.irap").exists()

    def load_surface_file(self, key: str, realization: int) -> RegularSurface:
        input_path = self.mount_point / f"realization-{realization}" / f"{key}.irap"
        if not input_path.exists():
            raise KeyError(f"No parameter: {key} in storage")
        surf = xtgeo.surface_from_file(input_path, fformat="irap_ascii")
        return surf

    def load_surface_data(self, key: str, realizations: List[int]) -> Any:
        result = []
        for realization in realizations:
            surf = self.load_surface_file(key, realization)
            result.append(surf.get_values1d(order="F"))
        return np.stack(result).T

    def has_parameters(self) -> bool:
        """
        Checks if a parameter file has been created
        """
        realization_folders = list(self.mount_point.glob("realization-*"))
        if not realization_folders:
            return False
        return (realization_folders[0] / "gen-kw.nc").exists()

    def load_gen_kw_realization(
        self, key: str, realization: int
    ) -> Tuple[npt.NDArray[np.double], List[str]]:
        input_file = self.mount_point / f"realization-{realization}" / "gen-kw.nc"
        if not input_file.exists():
            raise KeyError(f"Unable to load GEN_KW for key: {key}")
        with xr.open_dataset(input_file, engine="scipy") as ds_disk:
            np_data = ds_disk[key].to_numpy()
            keys = list(ds_disk[key][f"{key}_keys"].values)

        return np_data, keys

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

    def load_field(self, key: str, realizations: List[int]) -> npt.NDArray[np.double]:
        result: Optional[npt.NDArray[np.double]] = None
        for index, realization in enumerate(realizations):
            input_path = self.mount_point / f"realization-{realization}"
            if not input_path.exists():
                raise KeyError(
                    f"Unable to load FIELD for key: {key}, realization: {realization}"
                )
            data = np.load(input_path / f"{key}.npy", mmap_mode="r")
            data = data[np.isfinite(data)]

            if result is None:
                result = np.empty((len(realizations), data.size), dtype=np.double)
            elif data.size != result.shape[1]:
                raise ValueError(f"Data size mismatch for realization {realization}")

            result[index] = data

        if result is None:
            raise ConfigValidationError(
                "No realizations found when trying to load field"
            )
        return result.T

    def field_has_data(self, key: str, realization: int) -> bool:
        path = self.mount_point / f"realization-{realization}/{key}.npy"
        return path.exists()

    def field_has_info(self, key: str) -> bool:
        return key in self.experiment.parameter_info

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
        data = np.load(data_path / f"{key}.npy")
        data = field_transform(data, transform_name=info["output_transformation"])
        data = _field_truncate(
            data,
            info["truncation_min"],
            info["truncation_max"],
        )

        save_field(
            data,
            key,
            self.experiment.grid_path,
            Shape(info["nx"], info["ny"], info["nz"]),
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

    def save_ext_param(
        self, key: str, realization: int, data: Dict[str, Dict[str, Any]]
    ) -> None:
        output_path = self.mount_point / f"realization-{realization}"
        Path.mkdir(output_path, exist_ok=True)
        with open(output_path / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

    def save_surface_file(self, key: str, realization: int, file_name: str) -> None:
        output_path = self.mount_point / f"realization-{realization}"
        Path.mkdir(output_path, exist_ok=True)
        surf = xtgeo.surface_from_file(file_name, fformat="irap_ascii")
        surf.to_file(output_path / f"{key}.irap", fformat="irap_ascii")
        self.update_realization_state(
            realization,
            [RealizationStateEnum.STATE_UNDEFINED],
            RealizationStateEnum.STATE_INITIALIZED,
        )

    def save_surface_data(
        self,
        key: str,
        realization: int,
        data: npt.NDArray[np.double],
    ) -> None:
        output_path = self.mount_point / f"realization-{realization}"
        Path.mkdir(output_path, exist_ok=True)
        param_info = self.experiment.parameter_info[key]
        surf = xtgeo.RegularSurface(
            ncol=param_info["ncol"],
            nrow=param_info["nrow"],
            xinc=param_info["xinc"],
            yinc=param_info["yinc"],
            xori=param_info["xori"],
            yori=param_info["yori"],
            yflip=param_info["yflip"],
            rotation=param_info["rotation"],
            values=data,
        )
        surf.to_file(output_path / f"{key}.irap", fformat="irap_ascii")
        self.update_realization_state(
            realization,
            [RealizationStateEnum.STATE_UNDEFINED],
            RealizationStateEnum.STATE_INITIALIZED,
        )

    def copy_from_case(
        self, other: LocalEnsembleAccessor, nodes: List[str], active: List[bool]
    ) -> None:
        """
        This copies parameters from self into other, checking if nodes exists
        in self before performing copy.
        """
        self._copy_parameter_files(other, nodes, [i for i, b in enumerate(active) if b])

    def _copy_parameter_files(
        self,
        to: LocalEnsembleAccessor,
        parameter_keys: List[str],
        realizations: List[int],
    ) -> None:
        """
        Copies selected parameter files from one storage to another.
        Filters on realization and parameter keys
        """

        for f in ["gen-kw-priors.json"]:
            if not (self.mount_point / f).exists():
                continue
            shutil.copy(
                src=self.mount_point / f,
                dst=to.mount_point / f,
            )

        for realization_folder in self.mount_point.glob("realization-*"):
            files_to_copy = []
            realization = int(str(realization_folder).rsplit("-", maxsplit=1)[-1])
            if realization in realizations:
                for parameter_file in realization_folder.iterdir():
                    base_name = str(parameter_file.stem)
                    if (
                        base_name in parameter_keys
                        or parameter_file.name == "gen-kw.nc"
                    ):
                        files_to_copy.append(parameter_file.name)

            if not files_to_copy:
                continue

            Path.mkdir(to.mount_point / realization_folder.stem)
            for f in files_to_copy:
                shutil.copy(
                    src=self.mount_point / realization_folder.stem / f,
                    dst=to.mount_point / realization_folder.stem / f,
                )

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

    def save_gen_kw(
        self,
        parameter_name: str,
        parameter_keys: List[str],
        realization: int,
        data: npt.NDArray[np.double],
    ) -> None:
        ds = xr.Dataset(
            {parameter_name: ((f"{parameter_name}_keys"), data)},
            coords={f"{parameter_name}_keys": parameter_keys},
        )
        output_path = self.mount_point / f"realization-{realization}"
        Path.mkdir(output_path, exist_ok=True)
        mode: Literal["a", "w"] = "a" if Path.exists(output_path / "gen-kw.nc") else "w"
        ds.to_netcdf(output_path / "gen-kw.nc", mode=mode, engine="scipy")
        self.update_realization_state(
            realization,
            [RealizationStateEnum.STATE_UNDEFINED],
            RealizationStateEnum.STATE_INITIALIZED,
        )

    def save_response(self, group: str, data: xr.Dataset, realization: int) -> None:
        data = data.expand_dims({"realization": [realization]})
        output_path = self.mount_point / f"realization-{realization}"
        Path.mkdir(output_path, parents=True, exist_ok=True)

        data.to_netcdf(output_path / f"{group}.nc", engine="scipy")

    def save_field(
        self,
        parameter_name: str,
        realization: int,
        data: Union[
            npt.NDArray[np.double], np.ma.MaskedArray[Any, np.dtype[np.double]]
        ],
    ) -> None:
        output_path = self.mount_point / f"realization-{realization}"
        Path.mkdir(output_path, exist_ok=True)
        if not np.ma.isMaskedArray(data):  # type: ignore
            grid_path = self.experiment.grid_path
            if grid_path is None:
                raise ConfigValidationError(
                    f"Missing path to grid file for realization-{realization}"
                )
            mask, shape = read_mask(grid_path)
            if mask is not None:
                data_full = np.full_like(mask, np.nan, dtype=np.double)
                np.place(data_full, np.logical_not(mask), data)
                data = np.ma.MaskedArray(
                    data_full, mask, fill_value=np.nan
                )  # type: ignore
            else:
                data = np.ma.MaskedArray(data.reshape(shape))  # type: ignore

        np.save(f"{output_path}/{parameter_name}", data.filled(np.nan))  # type: ignore
        self.update_realization_state(
            realization,
            [RealizationStateEnum.STATE_UNDEFINED],
            RealizationStateEnum.STATE_INITIALIZED,
        )
