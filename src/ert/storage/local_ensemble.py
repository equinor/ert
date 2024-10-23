from __future__ import annotations

import contextlib
import logging
import os
from datetime import datetime
from functools import lru_cache, reduce
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from uuid import UUID

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel
from typing_extensions import deprecated

from ert.config.gen_kw_config import GenKwConfig
from ert.storage.mode import BaseMode, Mode, require_write

from .realization_storage_state import RealizationStorageState

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage.local_experiment import LocalExperiment
    from ert.storage.local_storage import LocalStorage

logger = logging.getLogger(__name__)

import polars
from polars.exceptions import ColumnNotFoundError


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
    ):
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

        @lru_cache(maxsize=None)
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
        prior_ensemble_id: Optional[UUID],
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

        storage._write_transaction(
            path / "index.json", index.model_dump_json().encode("utf-8")
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
    def parent(self) -> Optional[UUID]:
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
                e
                not in [
                    RealizationStorageState.PARENT_FAILURE,
                    RealizationStorageState.LOAD_FAILURE,
                ]
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
                self._parameters_exist_for_realization(i)
                for i in range(self.ensemble_size)
            ]
        )

    def get_realization_mask_with_responses(
        self, key: Optional[str] = None
    ) -> npt.NDArray[np.bool_]:
        """
        Mask array indicating realizations with associated responses.

        Parameters
        ----------
        key : str, optional
            Response key to filter realizations. If None, all responses are considered.

        Returns
        -------
        masks : ndarray of bool
            Boolean array where True means responses are associated.
        """

        return np.array(
            [
                self._responses_exist_for_realization(i, key)
                for i in range(self.ensemble_size)
            ]
        )

    def _parameters_exist_for_realization(self, realization: int) -> bool:
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
            (path / (_escape_filename(parameter) + ".nc")).exists()
            for parameter in self.experiment.parameter_configuration
        )

    def _responses_exist_for_realization(
        self, realization: int, key: Optional[str] = None
    ) -> bool:
        """
        Returns true if there are responses in the experiment and they have
        all been saved in the ensemble

        Parameters
        ----------
        realization : int
            Realization index.
        key : str, optional
            Response key to filter realizations. If None, all responses are considered.

        Returns
        -------
        exists : bool
            True if responses exist for realization.
        """

        if not self.experiment.response_configuration:
            return True
        path = self._realization_dir(realization)

        # Note: potential performance bottleneck,
        # should be improved greatly by having statemap.
        # Should also be faster due to lru cache on load_responses
        def _has_response(_response_type: str) -> bool:
            if (path / f"{_response_type}.parquet").exists():
                return realization in self.load_responses(response_type)["realization"]

            if (self.mount_point / f"{_response_type}.parquet").exists():
                return (
                    realization
                    in self.load_responses(
                        _response_type, tuple(range(self.ensemble_size))
                    )["realization"]
                )

        if key:
            response_type = self.experiment.response_key_to_response_type.get(key, key)
            return _has_response(response_type)

        return all(
            _has_response(response_type)
            for response_type in self.experiment.response_configuration
        )

    def is_initalized(self) -> List[int]:
        """
        Return the realization numbers where all parameters are internalized. In
        cases where there are parameters which are read from the forward model, an
        ensemble is considered initialized if all other parameters are present

        Returns
        -------
        exists : List[int]
            Returns the realization numbers with parameters
        """

        return [
            i
            for i in range(self.ensemble_size)
            if all(
                (
                    self._realization_dir(i)
                    / (_escape_filename(parameter.name) + ".nc")
                ).exists()
                for parameter in self.experiment.parameter_configuration.values()
                if not parameter.forward_init
            )
        ]

    def has_data(self) -> List[int]:
        """
        Return the realization numbers where all responses are internalized

        Returns
        -------
        exists : List[int]
            Returns the realization numbers with responses
        """

        # First check for combined
        realizations_with_responses_combined = {
            response: set(
                polars.read_parquet(self._path / f"{response}.parquet")[
                    "realization"
                ].unique()
            )
            if (self._path / f"{response}.parquet").exists()
            else {}
            for response in self.experiment.response_configuration
        }

        reals_with_all_responses_from_combined = reduce(
            set.intersection, realizations_with_responses_combined.values()
        )

        realizations_missing_responses = reals_with_all_responses_from_combined - set(
            range(self.ensemble_size)
        )

        if not realizations_missing_responses:
            return list(reals_with_all_responses_from_combined)

        # If not combined, fallback to checking individual folders
        # We assume that there is no state where a combine was invoked before
        # all responses were saved
        return [
            i
            for i in range(self.ensemble_size)
            if all(
                (self._realization_dir(i) / f"{response}.parquet").exists()
                for response in self.experiment.response_configuration
            )
        ]

    def realizations_initialized(self, realizations: List[int]) -> bool:
        """
        Check if specified realizations are initialized.

        Parameters
        ----------
        realizations : list of int
            List of realization indices.

        Returns
        -------
        initialized : bool
            True if all realizations are initialized.
        """

        responses = self.get_realization_mask_with_responses()
        parameters = self.get_realization_mask_with_parameters()

        if len(responses) == 0 and len(parameters) == 0:
            return False

        return all((responses[real] or parameters[real]) for real in realizations)

    def get_realization_list_with_responses(
        self, key: Optional[str] = None
    ) -> List[int]:
        """
        List of realization indices with associated responses.

        Parameters
        ----------
        key : str, optional
            Response key to filter realizations. If None, all responses are considered.

        Returns
        -------
        realizations : list of int
            List of realization indices with associated responses.
        """

        mask = self.get_realization_mask_with_responses(key)
        return np.where(mask)[0].tolist()

    def set_failure(
        self,
        realization: int,
        failure_type: RealizationStorageState,
        message: Optional[str] = None,
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
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        error = _Failure(
            type=failure_type, message=message if message else "", time=datetime.now()
        )
        self._storage._write_transaction(
            filename, error.model_dump_json().encode("utf-8")
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

    def get_failure(self, realization: int) -> Optional[_Failure]:
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

    def get_ensemble_state(self) -> List[RealizationStorageState]:
        """
        Retrieve the state of each realization within ensemble.

        Returns
        -------
        states : list of RealizationStorageState
            List of realization states.
        """

        def _find_state(realization: int) -> RealizationStorageState:
            if self.has_failure(realization):
                failure = self.get_failure(realization)
                assert failure
                return failure.type
            if self._responses_exist_for_realization(realization):
                return RealizationStorageState.HAS_DATA
            if self._parameters_exist_for_realization(realization):
                return RealizationStorageState.INITIALIZED
            else:
                return RealizationStorageState.UNDEFINED

        return [_find_state(i) for i in range(self.ensemble_size)]

    def get_summary_keyset(self) -> List[str]:
        """
        Find the first folder with summary data then load the
        summary keys from this.

        Returns
        -------
        keys : list of str
            List of summary keys.
        """

        try:
            summary_data = self.load_responses(
                "summary",
                tuple(self.get_realization_list_with_responses("summary")),
            )

            return sorted(summary_data["response_key"].unique().to_list())
        except (ValueError, KeyError, ColumnNotFoundError):
            return []

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
        realizations: Union[int, npt.NDArray[np.int_], None],
    ) -> xr.Dataset:
        if isinstance(realizations, int):
            return self._load_single_dataset(group, realizations).isel(
                realizations=0, drop=True
            )

        if realizations is None:
            datasets = [
                xr.open_dataset(p, engine="scipy")
                for p in sorted(
                    self.mount_point.glob(f"realization-*/{_escape_filename(group)}.nc")
                )
            ]
        else:
            datasets = [self._load_single_dataset(group, i) for i in realizations]
        return xr.combine_nested(datasets, concat_dim="realizations")

    def load_parameters(
        self, group: str, realizations: Union[int, npt.NDArray[np.int_], None] = None
    ) -> xr.Dataset:
        """
        Load parameters for group and realizations into xarray Dataset.

        Parameters
        ----------
        group : str
            Name of parameter group to load.
        realizations : {int, ndarray of int}, optional
            Realization indices to load. If None, all realizations are loaded.

        Returns
        -------
        parameters : Dataset
            Loaded xarray Dataset with parameters.
        """

        return self._load_dataset(group, realizations)

    def load_cross_correlations(self) -> xr.Dataset:
        input_path = self.mount_point / "corr_XY.nc"

        if not input_path.exists():
            raise FileNotFoundError(
                f"No cross-correlation data available at '{input_path}'. Make sure to run the update with "
                "Adaptive Localization enabled."
            )
        logger.info("Loading cross correlations")
        return xr.open_dataset(input_path, engine="scipy")

    @require_write
    def save_observation_scaling_factors(self, dataset: polars.DataFrame) -> None:
        self._storage._to_parquet_transaction(
            self.mount_point / "observation_scaling_factors.parquet", dataset
        )

    def load_observation_scaling_factors(
        self,
    ) -> Optional[polars.DataFrame]:
        ds_path = self.mount_point / "observation_scaling_factors.parquet"
        if ds_path.exists():
            return polars.read_parquet(ds_path)

        return None

    @require_write
    def save_cross_correlations(
        self,
        cross_correlations: npt.NDArray[np.float64],
        param_group: str,
        parameter_names: List[str],
    ) -> None:
        data_vars = {
            param_group: xr.DataArray(
                data=cross_correlations,
                dims=["parameter", "response"],
                coords={"parameter": parameter_names},
            )
        }
        dataset = xr.Dataset(data_vars)
        file_path = os.path.join(self.mount_point, "corr_XY.nc")
        self._storage._to_netcdf_transaction(file_path, dataset)

    @lru_cache  # noqa: B019
    def load_responses(self, key: str, realizations: Tuple[int]) -> polars.DataFrame:
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

        response_file = f"{response_type}.parquet"

        if os.path.exists(self.mount_point / response_file):
            df = polars.read_parquet(self.mount_point / response_file)
            if select_key:
                df = df.filter(polars.col("response_key") == key)

            if set(realizations) != {range(self.ensemble_size)}:
                df = df.filter(polars.col("realization").is_in(realizations))

            return df

        loaded = []
        for realization in realizations:
            input_path = self._realization_dir(realization) / f"{response_type}.parquet"
            if not input_path.exists():
                raise KeyError(f"No response for key {key}, realization: {realization}")
            df = polars.read_parquet(input_path)

            if select_key:
                df = df.filter(polars.col("response_key") == key)

            loaded.append(df)

        return polars.concat(loaded) if loaded else polars.DataFrame()

    @deprecated("Use load_responses")
    def load_all_summary_data(
        self,
        keys: Optional[List[str]] = None,
        realization_index: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load all summary data for realizations into pandas DataFrame.

        Parameters
        ----------
        keys : list of str, optional
            List of keys to load. If None, all keys are loaded.
        realization_index : int, optional

        Returns
        -------
        summary_data : DataFrame
            Loaded pandas DataFrame with summary data.
        """

        realizations = self.get_realization_list_with_responses()
        if realization_index is not None:
            if realization_index not in realizations:
                raise IndexError(f"No such realization {realization_index}")
            realizations = [realization_index]

        summary_keys = self.get_summary_keyset()

        try:
            df_pl = self.load_responses("summary", tuple(realizations))

        except (ValueError, KeyError):
            return pd.DataFrame()
        df_pl = df_pl.pivot(
            on="response_key", index=["realization", "time"], sort_columns=True
        )
        df_pl = df_pl.rename({"time": "Date", "realization": "Realization"})

        df_pandas = (
            df_pl.to_pandas()
            .set_index(["Realization", "Date"])
            .sort_values(by=["Date", "Realization"])
        )

        if keys:
            summary_keys = sorted(
                [key for key in keys if key in summary_keys]
            )  # ignore keys that doesn't exist
            return df_pandas[summary_keys]

        return df_pandas

    def load_all_gen_kw_data(
        self,
        group: Optional[str] = None,
        realization_index: Optional[int] = None,
    ) -> pd.DataFrame:
        """Loads scalar parameters (GEN_KWs) into a pandas DataFrame
        with columns <PARAMETER_GROUP>:<PARAMETER_NAME> and
        "Realization" as index.

        Parameters
        ----------
        group : str, optional
            Name of parameter group to load.
        relization_index : int, optional
            The realization to load.

        Returns
        -------
        data : DataFrame
            A pandas DataFrame containing the GEN_KW data.

        Notes
        -----
        Any provided keys that are not gen_kw will be ignored.
        """
        if realization_index is not None:
            realizations = np.array([realization_index])
        else:
            ens_mask = (
                self.get_realization_mask_with_responses()
                + self.get_realization_mask_with_parameters()
            )
            realizations = np.flatnonzero(ens_mask)

        dataframes = []
        gen_kws = [
            config
            for config in self.experiment.parameter_configuration.values()
            if isinstance(config, GenKwConfig)
        ]
        if group:
            gen_kws = [config for config in gen_kws if config.name == group]
        for key in gen_kws:
            with contextlib.suppress(KeyError):
                da = self.load_parameters(key.name, realizations)["transformed_values"]
                assert isinstance(da, xr.DataArray)
                da["names"] = np.char.add(f"{key.name}:", da["names"].astype(np.str_))
                df = da.to_dataframe().unstack(level="names")
                df.columns = df.columns.droplevel()
                for parameter in df.columns:
                    if key.shouldUseLogScale(parameter.split(":")[1]):
                        df[f"LOG10_{parameter}"] = np.log10(df[parameter])
                dataframes.append(df)
        if not dataframes:
            return pd.DataFrame()

        dataframe = pd.concat(dataframes, axis=1)
        dataframe.columns.name = None
        dataframe.index.name = "Realization"

        return dataframe.sort_index(axis=1)

    @require_write
    def save_parameters(
        self,
        group: str,
        realization: int,
        dataset: xr.Dataset,
    ) -> None:
        """
        Saves the provided dataset under a parameter group and realization index

        Parameters
        ----------
        group : str
            Parameter group name for saving dataset.

        realization : int
            Realization index for saving group.

        dataset : Dataset
            Dataset to save. It must contain a variable named 'values'
            which will be used when flattening out the parameters into
            a 1d-vector.
        """

        if "values" not in dataset.variables:
            raise ValueError(
                f"Dataset for parameter group '{group}' "
                f"must contain a 'values' variable"
            )

        if dataset["values"].size == 0:
            raise ValueError(
                f"Parameters {group} are empty. Cannot proceed with saving to storage."
            )

        if dataset["values"].ndim >= 2 and dataset["values"].values.dtype == "float64":
            logger.warning(
                "Dataset uses 'float64' for fields/surfaces. Use 'float32' to save memory."
            )

        if group not in self.experiment.parameter_configuration:
            raise ValueError(f"{group} is not registered to the experiment.")

        path = self._realization_dir(realization) / f"{_escape_filename(group)}.nc"
        path.parent.mkdir(exist_ok=True)

        self._storage._to_netcdf_transaction(
            path, dataset.expand_dims(realizations=[realization])
        )

    @require_write
    def save_response(
        self, response_type: str, data: polars.DataFrame, realization: int
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
                f"Responses {response_type} are empty. Cannot proceed with saving to storage."
            )

        if "realization" not in data.columns:
            data.insert_column(
                0,
                polars.Series(
                    "realization", np.full(len(data), realization), dtype=polars.UInt16
                ),
            )

        output_path = self._realization_dir(realization)
        Path.mkdir(output_path, parents=True, exist_ok=True)

        self._storage._to_parquet_transaction(
            output_path / f"{response_type}.parquet", data
        )

    def calculate_std_dev_for_parameter(self, parameter_group: str) -> xr.Dataset:
        if parameter_group not in self.experiment.parameter_configuration:
            raise ValueError(f"{parameter_group} is not registered to the experiment.")

        ds = self.load_parameters(parameter_group)
        return ds.std("realizations")

    def get_parameter_state(
        self, realization: int
    ) -> Dict[str, RealizationStorageState]:
        path = self._realization_dir(realization)
        return {
            e: RealizationStorageState.INITIALIZED
            if (path / (_escape_filename(e) + ".nc")).exists()
            else RealizationStorageState.UNDEFINED
            for e in self.experiment.parameter_configuration
        }

    def get_response_state(
        self, realization: int
    ) -> Dict[str, RealizationStorageState]:
        path = self._realization_dir(realization)
        return {
            e: RealizationStorageState.HAS_DATA
            if (path / f"{e}.parquet").exists()
            else RealizationStorageState.UNDEFINED
            for e in self.experiment.response_configuration
        }

    def combine_responses(self, response_type: Optional[str] = None) -> None:
        if response_type is None:
            for _response_type in self.experiment.response_configuration:
                self.combine_responses(_response_type)

            return None

        if response_type not in self.experiment.response_configuration:
            # 2do make test for this
            raise KeyError(
                f"Trying to combine responses for unknown response type {response_type}"
            )

        for _response_type in self.experiment.response_configuration:
            response_file = f"{response_type}.parquet"

            reals_with_response = self.get_realization_list_with_responses(
                response_type
            )

            if not os.path.exists(self.mount_point / response_file):
                combined = self.load_responses(
                    response_type, tuple(reals_with_response)
                )
            else:
                # Some files may be newer
                combined = polars.read_parquet(self.mount_point / response_file)

                to_write = []
                for real in range(self.ensemble_size):
                    if (self._realization_dir(real) / response_file).exists():
                        response_df = polars.read_parquet(
                            self._realization_dir(real) / response_file
                        )
                        to_write.append(response_df)

                if len(to_write) == 0:
                    continue

                df_to_write = polars.concat(to_write)
                joined = combined.join(
                    df_to_write, how="left", on=combined.columns[:-1], suffix="_new"
                )

                updated = joined.with_columns(
                    polars.col("values_new")
                    .fill_null(polars.col("values"))
                    .alias("values")
                ).select(combined.columns)

                combined = updated

            combined.write_parquet(self.mount_point / response_file)

            for real in reals_with_response:
                if os.path.exists(self._realization_dir(real) / response_file):
                    os.remove(self._realization_dir(real) / response_file)
