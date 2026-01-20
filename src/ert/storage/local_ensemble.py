from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from functools import cache, cached_property, lru_cache
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
import polars as pl
import resfo
import xarray as xr
from pydantic import BaseModel
from typing_extensions import TypedDict

from ert.config import ParameterCardinality, ParameterConfig, SummaryConfig
from ert.config.response_config import InvalidResponseFile
from ert.substitutions import substitute_runpath_name

from .load_status import LoadResult
from .mode import BaseMode, Mode, require_write
from .realization_storage_state import RealizationStorageState

if TYPE_CHECKING:
    import numpy.typing as npt

    from .local_experiment import LocalExperiment
    from .local_storage import LocalStorage

logger = logging.getLogger(__name__)


class EverestRealizationInfo(TypedDict):
    model_realization: int
    perturbation: int  # -1 means it stems from unperturbed controls


SCALAR_FILENAME = "SCALAR"


class _Index(BaseModel):
    id: UUID
    experiment_id: UUID
    ensemble_size: int
    iteration: int
    name: str
    prior_ensemble_id: UUID | None
    started_at: datetime
    everest_realization_info: dict[int, EverestRealizationInfo] | None = None


class _Failure(BaseModel):
    type: RealizationStorageState
    message: str
    time: datetime


def _escape_filename(filename: str) -> str:
    return filename.replace("%", "%25").replace("/", "%2F")


class LocalEnsemble(BaseMode):
    """
    Represents an ensemble within the local storage system of ERT.

    Manages multiple realizations of experiments, including different sets of
    parameters and responses.
    """

    def __init__(
        self,
        storage: LocalStorage,
        path: Path,
        mode: Mode,
    ) -> None:
        """
        Initialize a LocalEnsemble instance.

        Parameters
        ----------
        storage : LocalStorage
            Local storage instance.
        path : Path
            File system path to ensemble data.
        mode : Mode
            Access mode for the ensemble (read/write).
        """

        super().__init__(mode)
        self._storage = storage
        self._path = path
        self._index = _Index.model_validate_json(
            (path / "index.json").read_text(encoding="utf-8")
        )
        self._error_log_name = "error.json"

        @cache
        def create_realization_dir(realization: int) -> Path:
            return self._path / f"realization-{realization}"

        self._realization_dir = create_realization_dir

    @classmethod
    def create(
        cls,
        storage: LocalStorage,
        path: Path,
        uuid: UUID,
        *,
        ensemble_size: int,
        experiment_id: UUID,
        iteration: int = 0,
        name: str,
        prior_ensemble_id: UUID | None,
    ) -> LocalEnsemble:
        """
        Create a new ensemble in local storage.

        Parameters
        ----------
        storage : LocalStorage
            Local storage instance.
        path : Path
            File system path for ensemble data.
        uuid : UUID
            Unique identifier for the new ensemble.
        ensemble_size : int
            Number of realizations.
        experiment_id : UUID
            Identifier of associated experiment.
        iteration : int
            Iteration number of ensemble.
        name : str
            Name of ensemble.
        prior_ensemble_id : UUID, optional
            Identifier of prior ensemble.

        Returns
        -------
        local_ensemble : LocalEnsemble
            Instance of the newly created ensemble.
        """

        index = _Index(
            id=uuid,
            ensemble_size=ensemble_size,
            experiment_id=experiment_id,
            iteration=iteration,
            name=name,
            prior_ensemble_id=prior_ensemble_id,
            started_at=datetime.now(),
        )

        storage._write_transaction(
            path / "index.json", index.model_dump_json(indent=2).encode("utf-8")
        )

        return cls(storage, path, Mode.WRITE)

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
    def parent(self) -> UUID | None:
        return self._index.prior_ensemble_id

    @property
    def experiment(self) -> LocalExperiment:
        return self._storage.get_experiment(self.experiment_id)

    @property
    def relative_weights(self) -> str:
        return self._storage.get_experiment(self.experiment_id).relative_weights

    def get_realization_mask_without_failure(self) -> npt.NDArray[np.bool_]:
        """
        Mask array indicating realizations without any failure.

        Returns
        -------
        failures : ndarray of bool
            Boolean array where True means no failure.
        """

        return np.array(
            [
                not {
                    RealizationStorageState.FAILURE_IN_PARENT,
                    RealizationStorageState.FAILURE_IN_CURRENT,
                }.intersection(e)
                for e in self.get_ensemble_state()
            ]
        )

    def get_realization_mask_with_parameters(self) -> npt.NDArray[np.bool_]:
        """
        Mask array indicating realizations with associated parameters.

        Returns
        -------
        parameters : ndarray of bool
            Boolean array where True means parameters are associated.
        """

        return np.array(
            [
                bool({RealizationStorageState.PARAMETERS_LOADED}.intersection(state))
                for state in self.get_ensemble_state()
            ]
        )

    def get_realization_mask_with_responses(self) -> npt.NDArray[np.bool_]:
        """
        Get a boolean mask indicating which realizations have responses loaded.

        Returns
        -------
        ndarray of bool
            Boolean array where True means responses are loaded.
        """

        return np.array(
            [
                RealizationStorageState.RESPONSES_LOADED in state
                for state in self.get_ensemble_state()
            ]
        )

    @cached_property
    def _existing_scalars(self) -> dict[str, list[int]]:
        group_path = self.mount_point / f"{_escape_filename(SCALAR_FILENAME)}.parquet"
        genkw_mask: dict[str, list[int]] = {
            param_name: []
            for param_name, param in self.experiment.parameter_configuration.items()
            if param.cardinality
            == ParameterCardinality.multiple_configs_per_ensemble_dataset
        }
        if not group_path.exists():
            return genkw_mask
        df = pl.scan_parquet(group_path)
        cols = df.collect_schema().names()
        real = (
            df.select("realization")
            .unique()
            .collect()
            .get_column("realization")
            .to_list()
        )
        return {
            param: real
            for param in cols
            if param != "realization" and param in genkw_mask
        }

    def has_data(self) -> bool:
        """
        Check if the ensemble has any responses.

        Returns
        -------
        bool
            True if the ensemble has at least one realization with responses,
            False otherwise.
        """
        return len(self.get_realization_list_with_responses()) > 0

    def get_realization_list_with_responses(self) -> list[int]:
        """
        list of realization indices with associated responses.

        Parameters
        ----------
        key : str, optional
            Response key to filter realizations. If None, all responses are considered.

        Returns
        -------
        realizations : list of int
            list of realization indices with associated responses.
        """

        mask = self.get_realization_mask_with_responses()
        return np.where(mask)[0].tolist()

    def set_failure(
        self,
        realization: int,
        failure_type: RealizationStorageState,
        message: str | None = None,
    ) -> None:
        """
        Record a failure for a given realization in ensemble.

        Parameters
        ----------
        realization : int
            Index of realization.
        failure_type : RealizationStorageState
            Type of failure.
        message : str, optional
            Optional message describing the failure.
        """

        filename: Path = self._realization_dir(realization) / self._error_log_name
        error = _Failure(type=failure_type, message=message or "", time=datetime.now())
        if not filename.parent.parent.exists():
            logger.warning(
                f"Storage {filename.parent.parent} does not exist, "
                f"skipping writing {error} to {filename}"
            )
            return
        filename.parent.mkdir(exist_ok=True)
        self._storage._write_transaction(
            filename, error.model_dump_json(indent=2).encode("utf-8")
        )

    def unset_failure(
        self,
        realization: int,
    ) -> None:
        filename: Path = self._realization_dir(realization) / self._error_log_name
        if filename.exists():
            filename.unlink()

    def has_failure(self, realization: int) -> bool:
        """
        Check if given realization has a recorded failure.

        Parameters
        ----------
        realization : int
            Index of realization.

        Returns
        -------
        has_failure : bool
            True if realization has a recorded failure.
        """

        return (self._realization_dir(realization) / self._error_log_name).exists()

    def get_failure(self, realization: int) -> _Failure | None:
        """
        Retrieve failure information for a given realization, if any.

        Parameters
        ----------
        realization : int
            Index of realization.

        Returns
        -------
        failure : _Failure, optional
            Failure information if recorded, otherwise None.
        """

        if self.has_failure(realization):
            return _Failure.model_validate_json(
                (self._realization_dir(realization) / self._error_log_name).read_text(
                    encoding="utf-8"
                )
            )
        return None

    def refresh_ensemble_state(self) -> None:
        self.get_ensemble_state.cache_clear()
        if self._existing_scalars is not None:
            del self._existing_scalars
        self.get_ensemble_state()

    @lru_cache  # noqa: B019
    def get_ensemble_state(self) -> list[set[RealizationStorageState]]:
        """
        Retrieve the state of each realization within ensemble.

        Returns
        -------
        states : list of RealizationStorageState
            list of realization states.
        """

        response_configs = self.experiment.response_configuration
        existing_scalars = self._existing_scalars

        def _parameters_exist_for_realization(realization: int) -> bool:
            """
            Returns true if all parameters in the experiment have
            all been saved in the ensemble. If no parameters, return True

            Parameters
            ----------
            realization : int
                Realization index.

            Returns
            -------
            exists : bool
                True if parameters exist for realization.
            """
            if not self.experiment.parameter_configuration:
                return True
            path = self._realization_dir(realization)
            return all(
                (
                    parameter.name in existing_scalars
                    and realization in existing_scalars[parameter.name]
                )
                or ((path / (_escape_filename(parameter.name) + ".nc")).exists())
                for parameter in self.experiment.parameter_configuration.values()
            )

        def _responses_exist_for_realization(
            realization: int, key: str | None = None
        ) -> bool:
            """
            Returns true if there are responses in the experiment and they have
            all been saved in the ensemble

            Parameters
            ----------
            realization : int
                Realization index.
            key : str, optional
                Response key to filter realizations. If None, all
                responses are considered.

            Returns
            -------
            exists : bool
                True if responses exist for realization.
            """

            if not response_configs:
                return True
            path = self._realization_dir(realization)

            def _has_response(key_: str) -> bool:
                df_key = self.experiment.response_key_to_response_type.get(key_, key_)
                return (path / f"{df_key}.parquet").exists()

            if key:
                return _has_response(key)

            is_expecting_any_responses = any(
                bool(config.keys) for config in response_configs.values()
            )

            if not is_expecting_any_responses:
                return True

            non_empty_response_configs = [
                response
                for response, config in response_configs.items()
                if bool(config.keys)
            ]

            return all(
                _has_response(response) for response in non_empty_response_configs
            )

        def _find_state(realization: int) -> set[RealizationStorageState]:
            state = set()
            if self.has_failure(realization):
                failure = self.get_failure(realization)
                assert failure
                state.add(failure.type)
            if _responses_exist_for_realization(realization):
                state.add(RealizationStorageState.RESPONSES_LOADED)
            if _parameters_exist_for_realization(realization):
                state.add(RealizationStorageState.PARAMETERS_LOADED)

            if len(state) == 0:
                state.add(RealizationStorageState.UNDEFINED)

            return state

        return [_find_state(i) for i in range(self.ensemble_size)]

    def _load_single_dataset(
        self,
        group: str,
        realization: int,
    ) -> xr.Dataset:
        try:
            return xr.open_dataset(
                self.mount_point
                / f"realization-{realization}"
                / f"{_escape_filename(group)}.nc",
                engine="scipy",
            )
        except FileNotFoundError as e:
            raise KeyError(
                f"No dataset '{group}' in storage for realization {realization}"
            ) from e

    def _load_dataset(
        self,
        group: str,
        realizations: int | np.int64 | npt.NDArray[np.int_],
    ) -> xr.Dataset:
        if isinstance(realizations, int | np.int64):
            return self._load_single_dataset(group, int(realizations)).isel(
                realizations=0, drop=True
            )

        datasets = [self._load_single_dataset(group, int(i)) for i in realizations]
        return xr.combine_nested(datasets, concat_dim="realizations")

    def _load_parameters_lazy(
        self,
        group: str,
    ) -> pl.LazyFrame:
        """
        Lazy load genkw group with all realizations
        Parameters
        ----------
        group : str
            Name of parameter group to load.

        Returns
        -------
        parameters : pl.LazyFrame
            Loaded parameters.
        """
        group_path = self.mount_point / f"{_escape_filename(group)}.parquet"
        if not group_path.exists():
            raise KeyError(f"No {group} dataset in storage for ensemble {self.name}")
        df = pl.scan_parquet(group_path)
        return df

    def load_scalar_keys(
        self,
        keys: list[str] | None = None,
        realizations: int | npt.NDArray[np.int_] | None = None,
        transformed: bool = False,
    ) -> pl.DataFrame:
        if keys is None:
            keys = self.experiment.parameter_keys
        elif set(keys) - set(self.experiment.parameter_keys):
            missing = set(keys) - set(self.experiment.parameter_keys)
            raise KeyError(f"Parameters not registered to the experiment: {missing}")

        df_lazy = self._load_parameters_lazy(SCALAR_FILENAME)
        df_lazy = df_lazy.select(["realization", *keys])
        if realizations is not None:
            if isinstance(realizations, int):
                realizations = np.array([realizations])
            df_lazy = df_lazy.filter(pl.col("realization").is_in(realizations))
        df = df_lazy.collect(engine="streaming")
        if df.is_empty():
            raise IndexError(
                f"No matching realizations {realizations} found for {keys}"
            )

        if transformed:
            df = df.with_columns(
                [
                    pl.col(col)
                    .map_elements(
                        self.experiment.parameter_configuration[col].transform_data(),
                    )
                    .cast(df[col].dtype)
                    .alias(col)
                    for col in df.columns
                    if col != "realization"
                ]
            )
        return df

    def load_parameters(
        self,
        group: str,
        realizations: int | npt.NDArray[np.int_] | None = None,
        transformed: bool = False,
    ) -> xr.Dataset | pl.DataFrame:
        """
        Load parameters for group and realizations. If transformed is True,
        the parameters will be transformed using the parameter transformation
        otherwise it will return the raw values.

        """
        cfgs = [
            p
            for p in self.experiment.parameter_configuration.values()
            if group in {p.name, p.group_name}
        ]

        if not cfgs:
            raise KeyError(f"{group} is not registered to the experiment.")

        # if group refers to a group name, we expect the same cardinality
        cardinality = next(cfg.cardinality for cfg in cfgs)

        if cardinality == ParameterCardinality.multiple_configs_per_ensemble_dataset:
            return self.load_scalar_keys(
                [cfg.name for cfg in cfgs], realizations, transformed
            )
        return self._load_dataset(
            group,
            (
                realizations
                if realizations is not None
                else np.flatnonzero(self.get_realization_mask_with_parameters())
            ),
        )

    def load_parameters_numpy(
        self, group: str, realizations: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float64]:
        if group in self.experiment.parameter_configuration:
            config = self.experiment.parameter_configuration[group]
            return config.load_parameters(self, realizations)
        keys = [
            p.name
            for p in self.experiment.parameter_configuration.values()
            if group == p.group_name
            and p.cardinality
            == ParameterCardinality.multiple_configs_per_ensemble_dataset
        ]
        if keys:
            return (
                self.load_scalar_keys(keys, realizations)
                .drop("realization")
                .to_numpy()
                .T.copy()
            )
        raise KeyError(f"{group} is not registered to the experiment.")

    def save_parameters_numpy(
        self,
        parameters: npt.NDArray[np.float64],
        param_group: str,
        iens_active_index: npt.NDArray[np.int_],
    ) -> None:
        config_node = self.experiment.parameter_configuration[param_group]
        complete_df: pl.DataFrame | None = None
        with contextlib.suppress(KeyError):
            complete_df = self._load_parameters_lazy(SCALAR_FILENAME).collect(
                engine="streaming"
            )
        for real, ds in config_node.create_storage_datasets(
            parameters, iens_active_index
        ):
            if isinstance(ds, pl.DataFrame):
                if complete_df is None:
                    complete_df = ds
                else:
                    complete_df = (
                        complete_df.join(ds, on="realization", how="left")
                        .unique(subset=["realization"], keep="first")
                        .sort("realization")
                    )
            else:
                self.save_parameters(ds, config_node.name, real)

        group_path = self.mount_point / f"{_escape_filename(SCALAR_FILENAME)}.parquet"
        if complete_df is not None:
            self._storage._to_parquet_transaction(group_path, complete_df)

    def load_scalars(
        self, group: str | None = None, realizations: npt.NDArray[np.int_] | None = None
    ) -> pl.DataFrame:
        dataframes = []
        gen_kws = [
            p
            for p in self.experiment.parameter_configuration.values()
            if p.cardinality
            == ParameterCardinality.multiple_configs_per_ensemble_dataset
            and (group is None or p.group_name == group)
        ]

        for config in gen_kws:
            df = self.load_parameters(config.name, realizations, transformed=True)
            assert isinstance(df, pl.DataFrame)
            df = df.rename(
                {
                    col: f"{config.group_name}:{col}"
                    for col in df.columns
                    if col != "realization"
                }
            )
            dataframes.append(df)

        if not dataframes:
            return pl.DataFrame()

        return pl.concat(dataframes, how="align")

    @require_write
    def save_observation_scaling_factors(self, dataset: pl.DataFrame) -> None:
        self._storage._to_parquet_transaction(
            self.mount_point / "observation_scaling_factors.parquet", dataset
        )

    def load_observation_scaling_factors(
        self,
    ) -> pl.DataFrame | None:
        ds_path = self.mount_point / "observation_scaling_factors.parquet"
        if ds_path.exists():
            return pl.read_parquet(ds_path)

        return None

    @staticmethod
    def sample_parameter(
        parameter: ParameterConfig,
        real_nr: int,
        random_seed: int,
    ) -> pl.DataFrame:
        parameter_value = parameter.sample_value(
            str(random_seed),
            real_nr,
        )

        parameter_dict = {parameter.name: parameter_value[0]}
        parameter_dict["realization"] = real_nr
        return pl.DataFrame(
            parameter_dict,
            schema={parameter.name: pl.Float64, "realization": pl.Int64},
        )

    def load_responses(self, key: str, realizations: tuple[int, ...]) -> pl.DataFrame:
        """Load responses for key and realizations into xarray Dataset.

        For each given realization, response data is loaded from the NetCDF
        file whose filename matches the given key parameter.

        Parameters
        ----------
        key : str
            Response key to load.
        realizations : tuple of int
            Realization indices to load.

        Returns
        -------
        responses : DataFrame
            Loaded polars DataFrame with responses.
        """

        return self._load_responses_lazy(key, realizations).collect(engine="streaming")

    def _load_responses_lazy(
        self, key: str, realizations: tuple[int, ...]
    ) -> pl.LazyFrame:
        """Load responses for key and realizations into xarray Dataset.

        For each given realization, response data is loaded from the NetCDF
        file whose filename matches the given key parameter.

        Parameters
        ----------
        key : str
            Response key to load.
        realizations : tuple of int
            Realization indices to load.

        Returns
        -------
        responses : DataFrame
            Loaded polars DataFrame with responses.
        """

        select_key = False
        if key in self.experiment.response_configuration:
            response_type = key
        elif key not in self.experiment.response_key_to_response_type:
            raise ValueError(f"{key} is not a response")
        else:
            response_type = self.experiment.response_key_to_response_type[key]
            select_key = True

        loaded = []
        for realization in realizations:
            input_path = self._realization_dir(realization) / f"{response_type}.parquet"
            if not input_path.exists():
                raise KeyError(f"No response for key {key}, realization: {realization}")
            df = pl.scan_parquet(input_path)

            if select_key:
                df = df.filter(pl.col("response_key") == key)

            loaded.append(df)

        return pl.concat(loaded) if loaded else pl.DataFrame().lazy()

    @require_write
    def save_parameters(
        self,
        dataset: xr.Dataset | pl.DataFrame,
        group: str | None = None,
        realization: int | None = None,
    ) -> None:
        """
        Saves the provided dataset under a parameter group and realization index(es)

        """
        assert isinstance(dataset, (xr.Dataset | pl.DataFrame)), (
            f"Dataset must be either an xarray Dataset or polars Dataframe, "
            f"was '{type(dataset).__name__}'"
        )

        if isinstance(dataset, pl.DataFrame):
            if dataset.is_empty():
                raise ValueError("Parameters dataframe is empty.")
            allowed_cols = set(self.experiment.parameter_configuration) | {
                "realization"
            }
            actual_cols = set(dataset.columns)
            unexpected_cols = actual_cols - allowed_cols
            if unexpected_cols:
                raise KeyError(
                    f"Columns {', '.join(sorted(unexpected_cols))}"
                    " not in experiment parameters"
                )
            if "realization" not in dataset.columns:
                raise KeyError(
                    "DataFrame must contain a 'realization' column for"
                    " saving scalar parameters"
                )

            try:
                # since all realizations are saved in a single parquet file,
                # this makes sure that we only add / replace new data.
                df = self._load_parameters_lazy(SCALAR_FILENAME).collect(
                    engine="streaming"
                )
                df = df.drop(
                    [c for c in dataset.columns if c != "realization"], strict=False
                )
                df_full = (
                    df.join(dataset, on="realization", how="left")
                    .unique(subset=["realization"], keep="first")
                    .sort("realization")
                )
            except KeyError:
                df_full = dataset

            group_path = (
                self.mount_point / f"{_escape_filename(SCALAR_FILENAME)}.parquet"
            )
            self._storage._to_parquet_transaction(group_path, df_full)
            return

        assert group is not None, "Group must be provided for xarray Dataset"

        assert realization is not None, (
            "Realization must be provided for xarray Dataset"
        )
        if "values" not in dataset.variables:
            raise ValueError(
                f"Dataset for parameter group '{group}' "
                "must contain a 'values' variable"
            )
        if dataset["values"].size == 0:
            raise ValueError(
                f"Parameters {group} are empty. Cannot proceed with saving to storage."
            )

        path = self._realization_dir(realization) / f"{_escape_filename(group)}.nc"
        path.parent.mkdir(exist_ok=True)
        if "realizations" in dataset.dims:
            data_to_save = dataset.sel(realizations=[realization])
        else:
            data_to_save = dataset.expand_dims(realizations=[realization])
        self._storage._to_netcdf_transaction(path, data_to_save)

    @require_write
    def save_response(
        self, response_type: str, data: pl.DataFrame, realization: int
    ) -> None:
        """
        Save dataset as response under group and realization index.

        Parameters
        ----------
        response_type : str
            A name for the type of response stored, e.g., "summary, or "gen_data".
        realization : int
            Realization index for saving group.
        data : polars DataFrame
            polars DataFrame to save.
        """

        if "values" not in data.columns:
            raise ValueError(
                f"Dataset for response group '{response_type}' "
                f"must contain a 'values' variable"
            )

        if len(data) == 0:
            raise ValueError(
                f"Responses {response_type} are empty. "
                "Cannot proceed with saving to storage."
            )

        if "realization" not in data.columns:
            data.insert_column(
                0,
                pl.Series(
                    "realization", np.full(len(data), realization), dtype=pl.UInt16
                ),
            )

        output_path = self._realization_dir(realization)
        Path(output_path).mkdir(exist_ok=True)

        self._storage._to_parquet_transaction(
            output_path / f"{response_type}.parquet", data
        )

        if not self.experiment._has_finalized_response_keys(response_type):
            response_keys = data["response_key"].unique().to_list()
            self.experiment._update_response_keys(response_type, response_keys)

    def calculate_std_dev_for_parameter_group(
        self, parameter_group: str
    ) -> npt.NDArray[np.float64]:
        data = self.load_parameters(parameter_group)
        if isinstance(data, pl.DataFrame):
            return data.drop("realization").std().to_numpy().reshape(-1)
        return data.std("realizations")["values"].to_numpy()

    def get_parameter_state(
        self, realization: int
    ) -> dict[str, RealizationStorageState]:
        path = self._realization_dir(realization)
        existing_scalars = self._existing_scalars
        return {
            e: (
                RealizationStorageState.PARAMETERS_LOADED
                if (path / (_escape_filename(e) + ".nc")).exists()
                or (e in existing_scalars and realization in existing_scalars[e])
                else RealizationStorageState.UNDEFINED
            )
            for e in self.experiment.parameter_configuration
        }

    def get_response_state(
        self, realization: int
    ) -> dict[str, RealizationStorageState]:
        response_configs = self.experiment.response_configuration
        path = self._realization_dir(realization)
        return {
            e: (
                RealizationStorageState.RESPONSES_LOADED
                if (path / f"{e}.parquet").exists()
                else RealizationStorageState.UNDEFINED
            )
            for e in response_configs
        }

    def get_observations_and_responses(
        self,
        selected_observations: Iterable[str],
        iens_active_index: npt.NDArray[np.int_],
    ) -> pl.DataFrame:
        """Fetches and aligns selected observations with their
        corresponding simulated responses from an ensemble."""
        known_observations = self.experiment.observation_keys
        unknown_observations = [
            obs for obs in selected_observations if obs not in known_observations
        ]

        if unknown_observations:
            raise KeyError(
                f"Observations: {', '.join(unknown_observations)} not in experiment"
            )

        observations_by_type = self.experiment.observations

        with pl.StringCache():
            dfs_per_response_type = []
            for (
                response_type,
                response_cls,
            ) in self.experiment.response_configuration.items():
                if response_type not in observations_by_type:
                    continue

                observations_for_type = (
                    observations_by_type[response_type]
                    .filter(
                        pl.col("observation_key").is_in(list(selected_observations))
                    )
                    .with_columns(
                        [
                            pl.col("response_key")
                            .cast(pl.Categorical)
                            .alias("response_key")
                        ]
                    )
                )

                observed_cols = {
                    k: observations_for_type[k].unique()
                    for k in ["response_key", *response_cls.primary_key]
                }

                reals = iens_active_index.tolist()
                reals.sort()
                # too much memory to do it all at once, go per realization
                first_columns: pl.DataFrame | None = None
                realization_columns: list[pl.DataFrame] = []
                for real in reals:
                    responses = self._load_responses_lazy(
                        response_type, (real,)
                    ).with_columns(
                        [
                            pl.col("response_key")
                            .cast(pl.Categorical)
                            .alias("response_key")
                        ]
                    )

                    # Filter out responses without observations
                    for col, observed_values in observed_cols.items():
                        if col != "time":
                            responses = responses.filter(
                                pl.col(col).is_in(observed_values.implode())
                            )

                    pivoted = responses.collect(engine="streaming").pivot(
                        on="realization",
                        index=["response_key", *response_cls.primary_key],
                        values="values",
                        aggregate_function="mean",
                    )

                    if pivoted.is_empty():
                        # There are no responses for this realization,
                        # so we explicitly create a column of nans
                        # to represent this. We are basically saying that
                        # for this realization, each observation points
                        # to a NaN response.
                        joined = observations_for_type.with_columns(
                            pl.Series(
                                str(real),
                                [np.nan] * len(observations_for_type),
                                dtype=pl.Float32,
                            )
                        )
                    elif "time" in pivoted:
                        by_cols = [
                            "response_key",
                            *[k for k in response_cls.primary_key if k != "time"],
                        ]
                        joined = observations_for_type.sort(
                            by=[*by_cols, "time"]
                        ).join_asof(
                            pivoted.sort(by=[*by_cols, "time"]),
                            by=by_cols,
                            on="time",
                            check_sortedness=False,  # Ref: https://github.com/pola-rs/polars/issues/21693
                            strategy="nearest",
                            tolerance="1s",
                        )
                    else:
                        joined = observations_for_type.join(
                            pivoted,
                            how="left",
                            on=["response_key", *response_cls.primary_key],
                        )

                    joined = (
                        joined.with_columns(
                            pl.concat_str(
                                response_cls.primary_key, separator=", "
                            ).alias(
                                "__tmp_index_key__"
                                # Avoid potential collisions w/ primary key
                            )
                        )
                        .drop(response_cls.primary_key)
                        .rename({"__tmp_index_key__": "index"})
                    )

                    if first_columns is None:
                        # The "leftmost" index columns are not yet collected.
                        # They are the same for all iterations, and indexed the same
                        # because we do a left join for the observations.
                        # Hence, we select these columns only once.
                        first_columns = joined.select(
                            [
                                "response_key",
                                "index",
                                "observation_key",
                                "observations",
                                "std",
                                "location_x",
                                "location_y",
                                "location_range",
                            ]
                        )

                    realization_columns.append(joined.select(str(real)))

                if first_columns is None:
                    # Not a single realization had any responses to the
                    # observations. Hence, there is no need to include
                    # it in the dataset
                    continue

                dfs_per_response_type.append(
                    pl.concat([first_columns, *realization_columns], how="horizontal")
                )

            return pl.concat(dfs_per_response_type, how="vertical").with_columns(
                pl.col("response_key").cast(pl.String).alias("response_key")
            )

    @property
    def everest_realization_info(self) -> dict[int, EverestRealizationInfo] | None:
        return self._index.everest_realization_info

    def save_everest_realization_info(
        self, realization_info: dict[int, EverestRealizationInfo]
    ) -> None:
        if len(realization_info) != self.ensemble_size:
            raise ValueError(
                "Everest realization info must describe "
                "all realizations in the ensemble, got information "
                f"for realizations [{', '.join(map(str, realization_info))}]"
            )

        errors = []
        for ert_realization, info in realization_info.items():
            pert = info.get("perturbation")
            model_realization = info.get("model_realization")

            if pert is None or (pert < 0 and pert != -1):
                errors.append(
                    f"Invalid perturbation for "
                    f"ert realization: {ert_realization},"
                    f"expected -1 or a positive int"
                )

            if model_realization is None:
                errors.append(
                    f"Invalid model realization for ert realization {ert_realization}"
                )

        if errors:
            raise ValueError("Bad everest realization info: " + "\n".join(errors))

        self._index.everest_realization_info = realization_info
        self._storage._write_transaction(
            self._path / "index.json", self._index.model_dump_json().encode("utf-8")
        )


async def _read_parameters(
    run_path: str,
    realization: int,
    iteration: int,
    ensemble: LocalEnsemble,
) -> LoadResult:
    result = LoadResult.success()
    error_msg = ""
    parameter_configuration = ensemble.experiment.parameter_configuration.values()
    for config in parameter_configuration:
        if not config.forward_init:
            continue
        try:
            start_time = time.perf_counter()
            logger.debug(f"Starting to load parameter: {config.name}")
            ds = config.read_from_runpath(Path(run_path), realization, iteration)
            await asyncio.sleep(0)
            logger.debug(
                f"Loaded {config.name}",
                extra={"Time": f"{(time.perf_counter() - start_time):.4f}s"},
            )
            start_time = time.perf_counter()
            ensemble.save_parameters(ds, config.name, realization)
            await asyncio.sleep(0)
            logger.debug(
                f"Saved {config.name} to storage",
                extra={"Time": f"{(time.perf_counter() - start_time):.4f}s"},
            )
        except Exception as err:
            error_msg += str(err)
            result = LoadResult.failure(error_msg)
            logger.warning(
                "Failed to load parameters in storage "
                f"for realization {realization}: {err}"
            )
    return result


def _log_grid_contents(
    run_path: str, summary_config: SummaryConfig, iens: int, iter_: int
) -> None:
    try:
        filename = substitute_runpath_name(summary_config.input_files[0], iens, iter_)
        base, extension = os.path.splitext(filename)
        # Cut of extensions ".data", ".smspec" or ".unsmry"
        if extension.lower() in {".data", ".smspec", ".unsmry"}:
            filename = base
        for grid_file_components in filter(
            lambda fn: fn[0] == filename and fn[1].lower() in {".egrid", ".grid"},
            map(os.path.splitext, os.listdir(run_path)),
        ):
            grid_file = run_path + "/" + "".join(grid_file_components)
            keywords: Counter[str] = Counter()
            for entry in resfo.lazy_read(grid_file):
                kw = entry.read_keyword().strip()
                keywords[kw] += 1
                match kw:
                    case "FILEHEAD" | "GRIDHEAD":
                        arr = entry.read_array()
                        arr_len = 8 if kw == "FILEHEAD" else 33
                        arr_printout = (
                            "MESS"
                            if isinstance(arr, resfo.MessType)
                            else str(arr[0:arr_len])
                        )
                        logger.info(f"{grid_file} {kw} contains {arr_printout}")

            logger.info(f"{grid_file} contained keywords {sorted(keywords)}")

    except Exception as err:
        logger.error(f"Error while logging grid contents: {err}")


async def _write_responses_to_storage(
    run_path: str,
    realization: int,
    ensemble: LocalEnsemble,
) -> LoadResult:
    errors = []
    response_configs = ensemble.experiment.response_configuration.values()
    for config in response_configs:
        try:
            start_time = time.perf_counter()
            logger.debug(f"Starting to load response: {config.type}")
            try:
                if isinstance(config, SummaryConfig) and realization == 0:
                    _log_grid_contents(
                        run_path, config, realization, ensemble.iteration
                    )
                ds = config.read_from_file(run_path, realization, ensemble.iteration)
            except (FileNotFoundError, InvalidResponseFile) as err:
                errors.append(str(err))
                logger.warning(
                    f"Failed to read response from realization {realization}: {err}"
                )
                continue
            await asyncio.sleep(0)
            logger.debug(
                f"Loaded {config.type}",
                extra={"Time": f"{(time.perf_counter() - start_time):.4f}s"},
            )
            start_time = time.perf_counter()
            ensemble.save_response(config.type, ds, realization)
            await asyncio.sleep(0)
            logger.debug(
                f"Saved {config.type} to storage",
                extra={"Time": f"{(time.perf_counter() - start_time):.4f}s"},
            )
        except Exception as err:
            errors.append(str(err))
            logger.exception(
                "Unexpected exception while reading from runpath or "
                "writing response to storage "
                f"for realization {realization=}",
                exc_info=err,
            )
            continue

    if errors:
        return LoadResult.failure("\n".join(errors))
    return LoadResult.success()


async def load_realization_parameters_and_responses(
    run_path: str,
    realization: int,
    iter_: int,
    ensemble: LocalEnsemble,
) -> LoadResult:
    parameters_result = LoadResult.success()
    response_result = LoadResult.success()
    # We only read parameters after the prior, after that, ERT
    # handles parameters
    if iter_ == 0:
        parameters_result = await _read_parameters(
            run_path,
            realization,
            iter_,
            ensemble,
        )
    try:
        if parameters_result.successful:
            response_result = await _write_responses_to_storage(
                run_path,
                realization,
                ensemble,
            )
    except OSError as err:
        msg = (
            f"Failed to write responses to storage for realization {realization}, "
            f"failed with {err}"
        )
        logger.error(msg)
        parameters_result = LoadResult.failure(msg)
    except Exception as err:
        logger.exception(
            f"Failed to load results for realization {realization}",
            exc_info=err,
        )
        parameters_result = LoadResult.failure(
            f"Failed to load results for realization {realization}, failed with: {err}",
        )

    final_result = parameters_result
    try:
        if not response_result.successful:
            final_result = response_result
            ensemble.set_failure(
                realization,
                RealizationStorageState.FAILURE_IN_CURRENT,
                final_result.message,
            )
        elif ensemble.has_failure(realization):
            ensemble.unset_failure(realization)
    except OSError as err:
        logger.error(
            f"Failed to set realization state in storage for realization {realization},"
            f" failed with {err}"
        )

    return final_result


def load_parameters_and_responses_from_runpath(
    run_path_format: str,
    ensemble: LocalEnsemble,
    active_realizations: list[int],
) -> int:
    """Returns the number of loaded realizations"""
    pool = ThreadPool(processes=8)

    async_result = [
        (
            pool.apply_async(
                lambda *args: asyncio.run(
                    load_realization_parameters_and_responses(*args)
                ),
                (
                    substitute_runpath_name(run_path_format, realization, 0),
                    realization,
                    0,
                    ensemble,
                ),
            ),
            realization,
        )
        for realization in active_realizations
    ]

    loaded = 0
    for t, iens in async_result:
        (success, message) = t.get()

        if success:
            loaded += 1
        else:
            logger.error(f"Realization: {iens}, load failure: {message}")

    ensemble.refresh_ensemble_state()
    return loaded
