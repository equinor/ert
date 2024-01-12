from __future__ import annotations

import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from uuid import UUID

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel
from typing_extensions import deprecated

from ert.config.gen_data_config import GenDataConfig
from ert.config.gen_kw_config import GenKwConfig
from ert.config.response_config import ResponseConfig
from ert.config.summary_config import SummaryConfig

from .realization_storage_state import RealizationStorageState

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


class _Failure(BaseModel):
    type: RealizationStorageState
    message: str
    time: datetime


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
        self._error_log_name = "error.json"

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
    def experiment(self) -> Union[LocalExperimentReader, LocalExperimentAccessor]:
        return self._storage.get_experiment(self.experiment_id)

    def get_realization_mask_without_parent_failure(self) -> npt.NDArray[np.bool_]:
        return np.array(
            [
                (e != RealizationStorageState.PARENT_FAILURE)
                for e in self.get_ensemble_state()
            ]
        )

    def get_realization_mask_with_parameters(self) -> npt.NDArray[np.bool_]:
        return np.array([self._get_parameter(i) for i in range(self.ensemble_size)])

    def get_realization_mask_with_responses(self) -> npt.NDArray[np.bool_]:
        return np.array([self._get_response(i) for i in range(self.ensemble_size)])

    def _get_parameter(self, realization: int) -> bool:
        if not self.experiment.parameter_configuration:
            return False
        path = self.mount_point / f"realization-{realization}"
        return all(
            (path / f"{parameter}.nc").exists()
            for parameter in self.experiment.parameter_configuration
        )

    def _get_response(self, realization: int) -> bool:
        if not self.experiment.response_configuration:
            return False
        path = self.mount_point / f"realization-{realization}"
        return all(
            (path / f"{response}.nc").exists()
            for response in self._filter_response_configuration()
        )

    def _get_parameter_mask(self) -> List[bool]:
        return [
            all(
                [
                    (path / f"{parameter}.nc").exists()
                    for parameter in self.experiment.parameter_configuration
                ]
            )
            for path in sorted(list(self.mount_point.glob("realization-*")))
        ]

    def _get_response_mask(self) -> List[bool]:
        return [
            all(
                [
                    (path / f"{response}.nc").exists()
                    for response in self._filter_response_configuration()
                ]
            )
            for path in sorted(list(self.mount_point.glob("realization-*")))
        ]

    def _filter_response_configuration(self) -> Dict[str, ResponseConfig]:
        """
        Filter the response configuration removing summary responses with no keys. These produce no output file
        """
        return dict(
            filter(
                lambda x: not (isinstance(x[1], SummaryConfig) and not x[1].keys),
                self.experiment.response_configuration.items(),
            )
        )

    def is_initalized(self) -> bool:
        """
        Check that the ensemble has all parameters present in at least one realization
        """
        return any(self._get_parameter_mask())

    def has_data(self) -> bool:
        """
        Check that the ensemble has all responses present in at least one realization
        """
        return any(self._get_response_mask())

    def realizations_initialized(self, realizations: List[int]) -> bool:
        responses = self.get_realization_mask_with_responses()
        parameters = self.get_realization_mask_with_parameters()

        if len(responses) == 0 and len(parameters) == 0:
            return False

        return all((responses[real] or parameters[real]) for real in realizations)

    def get_realization_list_with_responses(self) -> List[int]:
        return [
            idx for idx, b in enumerate(self.get_realization_mask_with_responses()) if b
        ]

    def set_failure(
        self,
        realization: int,
        failure_type: RealizationStorageState,
        message: Optional[str] = None,
    ) -> None:
        filename: Path = (
            self._path / f"realization-{realization}" / self._error_log_name
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        error = _Failure(
            type=failure_type, message=message if message else "", time=datetime.now()
        )
        with open(filename, mode="w", encoding="utf-8") as f:
            print(error.model_dump_json(), file=f)

    def has_failure(self, realization: int) -> bool:
        return (
            self._path / f"realization-{realization}" / self._error_log_name
        ).exists()

    def get_failure(self, realization: int) -> Optional[_Failure]:
        if self.has_failure(realization):
            return _Failure.model_validate_json(
                (
                    self._path / f"realization-{realization}" / self._error_log_name
                ).read_text(encoding="utf-8")
            )
        return None

    def get_ensemble_state(self) -> List[RealizationStorageState]:
        def _find_state(realization: int) -> RealizationStorageState:
            if self.has_failure(realization):
                failure = self.get_failure(realization)
                assert failure
                return failure.type
            if self._get_response(realization):
                return RealizationStorageState.HAS_DATA
            if self._get_parameter(realization):
                return RealizationStorageState.INITIALIZED
            else:
                return RealizationStorageState.UNDEFINED

        return [_find_state(i) for i in range(self.ensemble_size)]

    def has_parameter_group(self, group: str) -> bool:
        param_group_file = self.mount_point / f"realization-0/{group}.nc"
        return param_group_file.exists()

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
        parameters = [
            config
            for config in self.experiment.parameter_configuration.values()
            if isinstance(config, GenKwConfig)
        ]
        for gen_kw_config in parameters:
            for keyword in [e.name for e in gen_kw_config.transfer_functions]:
                gen_kw_list.append(f"{gen_kw_config.name}:{keyword}")

                if gen_kw_config.shouldUseLogScale(keyword):
                    gen_kw_list.append(f"LOG10_{gen_kw_config.name}:{keyword}")

        return sorted(gen_kw_list, key=lambda k: k.lower())

    @deprecated("Use load_responses")
    def load_gen_data(
        self,
        key: str,
        report_step: int,
        realization_index: Optional[int] = None,
    ) -> pd.DataFrame:
        realizations = self.get_realization_list_with_responses()
        if realization_index is not None:
            if realization_index not in realizations:
                raise IndexError(f"No such realization {realization_index}")
            realizations = [realization_index]

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
        realizations = self.get_realization_list_with_responses()
        if realization_index is not None:
            if realization_index not in realizations:
                raise IndexError(f"No such realization {realization_index}")
            realizations = [realization_index]

        summary_keys = self.get_summary_keyset()

        try:
            df = self.load_responses("summary", tuple(realizations)).to_dataframe()
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
        ens_mask = (
            self.get_realization_mask_with_responses()
            + self.get_realization_mask_with_parameters()
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

    def save_response(self, group: str, data: xr.Dataset, realization: int) -> None:
        if "realization" not in data.dims:
            data = data.expand_dims({"realization": [realization]})
        output_path = self.mount_point / f"realization-{realization}"
        Path.mkdir(output_path, parents=True, exist_ok=True)

        data.to_netcdf(output_path / f"{group}.nc", engine="scipy")
