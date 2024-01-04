from __future__ import annotations

import json
import logging
from datetime import datetime
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from uuid import UUID

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel
from typing_extensions import deprecated

from ert.callbacks import forward_model_ok
from ert.config.gen_data_config import GenDataConfig
from ert.config.gen_kw_config import GenKwConfig
from ert.load_status import LoadResult, LoadStatus
from ert.storage.realization_storage_state import RealizationStorageState

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.run_arg import RunArg
    from ert.storage.local_experiment import (
        LocalExperimentAccessor,
        LocalExperimentReader,
    )
    from ert.storage.local_storage import LocalStorageAccessor, LocalStorageReader

logger = logging.getLogger(__name__)


def _load_realization(
    sim_fs: LocalEnsembleAccessor,
    realisation: int,
    run_args: List[RunArg],
) -> Tuple[LoadResult, int]:
    sim_fs.update_realization_storage_state(
        realisation,
        [RealizationStorageState.UNDEFINED],
        RealizationStorageState.INITIALIZED,
    )
    result = forward_model_ok(run_args[realisation])
    sim_fs.state_map[realisation] = (
        RealizationStorageState.HAS_DATA
        if result.status == LoadStatus.LOAD_SUCCESSFUL
        else RealizationStorageState.LOAD_FAILURE
    )
    return result, realisation


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
        self._index = _Index.model_validate_json(
            (path / "index.json").read_text(encoding="utf-8")
        )
        self._experiment_path = self._path / "experiment"

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
    def state_map(self) -> List[RealizationStorageState]:
        return self._state_map

    @property
    def experiment(self) -> Union[LocalExperimentReader, LocalExperimentAccessor]:
        return self._storage.get_experiment(self.experiment_id)

    @property
    def is_initalized(self) -> bool:
        return RealizationStorageState.INITIALIZED in self.state_map or self.has_data

    @property
    def has_data(self) -> bool:
        return RealizationStorageState.HAS_DATA in self.state_map

    def close(self) -> None:
        self.sync()

    def sync(self) -> None:
        pass

    def get_realization_mask_from_state(
        self, states: List[RealizationStorageState]
    ) -> npt.NDArray[np.bool_]:
        return np.array([s in states for s in self._state_map], dtype=bool)

    def _load_state_map(self) -> List[RealizationStorageState]:
        state_map_file = self._experiment_path / "state_map.json"
        if state_map_file.exists():
            with open(state_map_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [RealizationStorageState(v) for v in data["state_map"]]
        else:
            return [
                RealizationStorageState.UNDEFINED for _ in range(self.ensemble_size)
            ]

    def realizations_initialized(self, realizations: List[int]) -> bool:
        initialized_realizations = set(
            self.realization_list(RealizationStorageState.INITIALIZED)
        )
        return all(real in initialized_realizations for real in realizations)

    def has_parameter_group(self, group: str) -> bool:
        param_group_file = self.mount_point / f"realization-0/{group}.nc"
        return param_group_file.exists()

    def _filter_active_realizations(
        self, realization_index: Optional[int] = None
    ) -> List[int]:
        realizations = self.realization_list(RealizationStorageState.HAS_DATA)
        if realization_index is not None:
            if realization_index not in realizations:
                raise IndexError(f"No such realization {realization_index}")
            realizations = [realization_index]
        return realizations

    def realization_list(self, state: RealizationStorageState) -> List[int]:
        """
        Will return list of realizations with state == the specified state.
        """
        return [i for i, s in enumerate(self._state_map) if s == state]

    @deprecated("Check the experiment for registered responses")
    def get_summary_keyset(self) -> List[str]:
        """
        Find the first folder with summary data then load the
        summary keys from this.
        """
        for folder in list(self.mount_point.glob("realization-*")):
            if (folder / "summary.nc").exists():
                realization_nr = int(str(folder)[str(folder).rfind("-") + 1 :])
                response = self.load_responses("summary", (realization_nr,))
                return sorted(response["name"].values)
        return []

    def _get_gen_data_config(self, key: str) -> GenDataConfig:
        config = self.experiment.response_configuration[key]
        assert isinstance(config, GenDataConfig)
        return config

    @deprecated("Check the experiment for registered responses")
    def get_gen_data_keyset(self) -> List[str]:
        keylist = [
            k
            for k, v in self.experiment.response_info.items()
            if "_ert_kind" in v and v["_ert_kind"] == "GenDataConfig"
        ]

        gen_data_list = []
        for key in keylist:
            gen_data_config = self._get_gen_data_config(key)
            if gen_data_config.report_steps is None:
                gen_data_list.append(f"{key}@0")
            else:
                for report_step in gen_data_config.report_steps:
                    gen_data_list.append(f"{key}@{report_step}")
        return sorted(gen_data_list, key=lambda k: k.lower())

    @deprecated("Check the experiment for registered parameters")
    def get_gen_kw_keyset(self) -> List[str]:
        gen_kw_list = []
        for key in self.experiment.parameter_info:
            gen_kw_config = self.experiment.parameter_configuration[key]
            assert isinstance(gen_kw_config, GenKwConfig)

            for keyword in [e.name for e in gen_kw_config.transfer_functions]:
                gen_kw_list.append(f"{key}:{keyword}")

                if gen_kw_config.shouldUseLogScale(keyword):
                    gen_kw_list.append(f"LOG10_{key}:{keyword}")

        return sorted(gen_kw_list, key=lambda k: k.lower())

    def _load_single_dataset(
        self,
        group: str,
        realization: int,
    ) -> xr.Dataset:
        try:
            return xr.open_dataset(
                self.mount_point / f"realization-{realization}" / f"{group}.nc",
                engine="scipy",
            )
        except FileNotFoundError as e:
            raise KeyError(
                f"No dataset '{group}' in storage for realization {realization}"
            ) from e

    def _load_dataset(
        self,
        group: str,
        realizations: Union[int, npt.NDArray[np.int_], None],
    ) -> xr.Dataset:
        if isinstance(realizations, int):
            return self._load_single_dataset(group, realizations).isel(
                realizations=0, drop=True
            )

        if realizations is None:
            datasets = [
                xr.open_dataset(p, engine="scipy")
                for p in sorted(self.mount_point.glob(f"realization-*/{group}.nc"))
            ]
        else:
            datasets = [self._load_single_dataset(group, i) for i in realizations]
        return xr.combine_nested(datasets, "realizations")

    def load_parameters(
        self, group: str, realizations: Union[int, npt.NDArray[np.int_], None] = None
    ) -> xr.Dataset:
        return self._load_dataset(group, realizations)

    @lru_cache  # noqa: B019
    def load_responses(
        self, key: str, realizations: npt.NDArray[np.int_]
    ) -> xr.Dataset:
        if key not in self.experiment.response_configuration:
            raise ValueError(f"{key} is not a response")
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

    @deprecated("Use load_responses")
    def load_all_summary_data(
        self,
        keys: Optional[List[str]] = None,
        realization_index: Optional[int] = None,
    ) -> pd.DataFrame:
        summary_keys = self.get_summary_keyset()

        try:
            df = self.load_responses(
                "summary", tuple(self._filter_active_realizations(realization_index))
            ).to_dataframe()
        except (ValueError, KeyError):
            return pd.DataFrame()
        df = df.unstack(level="name")
        df.columns = [col[1] for col in df.columns.values]
        df.index = df.index.rename(
            {"time": "Date", "realization": "Realization"}
        ).reorder_levels(["Realization", "Date"])
        if keys:
            summary_keys = sorted(
                [key for key in keys if key in summary_keys]
            )  # ignore keys that doesn't exist
            return df[summary_keys]
        return df

    @deprecated("Use load_responses")
    def load_gen_data(
        self,
        key: str,
        report_step: int,
        realization_index: Optional[int] = None,
    ) -> pd.DataFrame:
        realizations = self._filter_active_realizations(realization_index)
        try:
            vals = self.load_responses(key, tuple(realizations)).sel(
                report_step=report_step, drop=True
            )
        except KeyError as e:
            raise KeyError(f"Missing response: {key}") from e
        index = pd.Index(vals.index.values, name="axis")
        return pd.DataFrame(
            data=vals["values"].values.reshape(len(vals.realization), -1).T,
            index=index,
            columns=realizations,
        )

    @deprecated("Use load_parameters")
    def load_all_gen_kw_data(
        self,
        group: Optional[str] = None,
        realization_index: Optional[int] = None,
    ) -> pd.DataFrame:
        """Loads all GEN_KW data into a DataFrame.

        This function retrieves GEN_KW data from the given ensemble reader.
        index and returns it in a pandas DataFrame.

        Args:
            ensemble: The ensemble reader from which to load the GEN_KW data.

        Returns:
            DataFrame: A pandas DataFrame containing the GEN_KW data.

        Raises:
            IndexError: If a non-existent realization index is provided.

        Note:
            Any provided keys that are not gen_kw will be ignored.
        """
        ens_mask = self.get_realization_mask_from_state(
            [
                RealizationStorageState.INITIALIZED,
                RealizationStorageState.HAS_DATA,
            ]
        )
        realizations = (
            np.array([realization_index])
            if realization_index is not None
            else np.flatnonzero(ens_mask)
        )

        dataframes = []
        gen_kws = [
            config
            for config in self.experiment.parameter_configuration.values()
            if isinstance(config, GenKwConfig)
        ]
        if group:
            gen_kws = [config for config in gen_kws if config.name == group]
        for key in gen_kws:
            try:
                ds = self.load_parameters(key.name, realizations)["transformed_values"]
                assert isinstance(ds, xr.DataArray)
                ds["names"] = np.char.add(f"{key.name}:", ds["names"].astype(np.str_))
                df = ds.to_dataframe().unstack(level="names")
                df.columns = df.columns.droplevel()
                for parameter in df.columns:
                    if key.shouldUseLogScale(parameter.split(":")[1]):
                        df[f"LOG10_{parameter}"] = np.log10(df[parameter])
                dataframes.append(df)
            except KeyError:
                pass
        if not dataframes:
            return pd.DataFrame()

        # Format the DataFrame in a way that old code expects it
        dataframe = pd.concat(dataframes, axis=1)
        dataframe.columns.name = None
        dataframe.index.name = "Realization"

        return dataframe.sort_index(axis=1)


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
            print(index.model_dump_json(), file=f)

        return cls(storage, path)

    def _save_state_map(self) -> None:
        state_map_file = self._experiment_path / "state_map.json"
        with open(state_map_file, "w", encoding="utf-8") as f:
            data = {"state_map": [v.value for v in self._state_map]}
            f.write(json.dumps(data))

    def update_realization_storage_state(
        self,
        realization: int,
        old_states: List[RealizationStorageState],
        new_state: RealizationStorageState,
    ) -> None:
        if self._state_map[realization] in old_states:
            self._state_map[realization] = new_state

    def sync(self) -> None:
        self._save_state_map()

    def load_from_run_path(
        self,
        ensemble_size: int,
        run_args: List[RunArg],
        active_realizations: List[bool],
    ) -> int:
        """Returns the number of loaded realizations"""
        pool = ThreadPool(processes=8)

        async_result = [
            pool.apply_async(
                _load_realization,
                (self, iens, run_args),
            )
            for iens in range(ensemble_size)
            if active_realizations[iens]
        ]

        loaded = 0
        for t in async_result:
            ((status, message), iens) = t.get()

            if status == LoadStatus.LOAD_SUCCESSFUL:
                loaded += 1
                self.state_map[iens] = RealizationStorageState.HAS_DATA
            else:
                logger.error(f"Realization: {iens}, load failure: {message}")

        return loaded

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
        if "values" not in dataset.variables:
            raise ValueError(
                f"Dataset for parameter group '{group}' "
                f"must contain a 'values' variable"
            )

        if group not in self.experiment.parameter_configuration:
            raise ValueError(f"{group} is not registered to the experiment.")

        path = self.mount_point / f"realization-{realization}" / f"{group}.nc"
        path.parent.mkdir(exist_ok=True)

        dataset.expand_dims(realizations=[realization]).to_netcdf(path, engine="scipy")
        self.update_realization_storage_state(
            realization,
            [
                RealizationStorageState.UNDEFINED,
                RealizationStorageState.LOAD_FAILURE,
            ],
            RealizationStorageState.INITIALIZED,
        )

    def save_response(self, group: str, data: xr.Dataset, realization: int) -> None:
        if "realization" not in data.dims:
            data = data.expand_dims({"realization": [realization]})
        output_path = self.mount_point / f"realization-{realization}"
        Path.mkdir(output_path, parents=True, exist_ok=True)

        data.to_netcdf(output_path / f"{group}.nc", engine="scipy")
