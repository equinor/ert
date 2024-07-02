from __future__ import annotations

import contextlib
import glob
import json
import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)
from uuid import UUID

import numpy as np
import pandas as pd
import xarray as xr
from pandas import DataFrame
from pydantic import BaseModel
from typing_extensions import deprecated

from ert.config.gen_kw_config import GenKwConfig
from ert.config.observations import ObservationsIndices
from ert.storage.mode import BaseMode, Mode, require_write

from ..config import GenDataConfig, ResponseTypes
from .ensure_correct_xr_coordinate_order import ensure_correct_coordinate_order
from .realization_storage_state import RealizationStorageState

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage.local_experiment import LocalExperiment
    from ert.storage.local_storage import LocalStorage

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


class ObservationsAndResponsesData:
    def __init__(self, observations_and_responses: pd.DataFrame) -> None:
        self._observations_and_responses = observations_and_responses

    def to_long_dataframe(self) -> pd.DataFrame:
        return self._observations_and_responses.set_index(
            ["name", "key_index"], verify_integrity=True
        )

    def index(self) -> npt.NDArray[np.str_]:
        """
        Extracts a ndarray with the shape (num_obs,).
        Each cell holds the observation primary key.
        """
        return self._observations_and_responses.loc[:, "key_index"].values

    def observation_keys(self) -> npt.NDArray[np.str_]:
        """
        Extracts a ndarray with the shape (num_obs,).
        Each cell holds the observation name.
        """
        return self._observations_and_responses.loc[:, "name"].values

    def errors(self) -> npt.NDArray[np.float64]:
        """
        Extracts a ndarray with the shape (num_obs,).
        Each cell holds the std. error of the observed value.
        """
        return self._observations_and_responses.loc[:, "STD"].values

    def observations(self) -> npt.NDArray[np.float64]:
        """
        Extracts a ndarray with the shape (num_obs,).
        Each cell holds the observed value.
        """
        return self._observations_and_responses.loc[:, "OBS"].values

    def responses(self) -> npt.NDArray[np.float64]:
        """
        Extracts a ndarray with the shape (num_obs, num_reals).
        Each cell holds the response value corresponding to the observation/realization
        indicated by the index.
        """
        return self._observations_and_responses.iloc[:, 4:].values


class RealizationState:
    def __init__(self) -> None:
        self._states: Dict[int, Set[Tuple[str, str, bool]]] = {}

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RealizationState):
            return self._states == other._states

        return False

    def clear_entry(self, realization: int, key: str) -> None:
        if realization not in self._states:
            return

        state_for_real = self._states[realization]

        to_remove = []
        for tup in state_for_real:
            _group, _key, _value = tup

            if key in {_group, _key}:
                to_remove.append(tup)

        for tup in to_remove:
            state_for_real.remove(tup)

    @staticmethod
    def from_file(path: Path) -> RealizationState:
        with open(path, "r") as f:
            return RealizationState.from_json(json.load(f))

    @staticmethod
    def from_json(data: Dict[str, List[Tuple[str, str, bool]]]) -> RealizationState:
        the_state = RealizationState()
        for realization, entries in data.items():
            the_state._states[int(realization)] = {
                (str(group), str(key), bool(value)) for group, key, value in entries
            }

        return the_state

    def _to_json(self) -> Dict[int, List[Tuple[str, str, bool]]]:
        return {i: list(states) for i, states in self._states.items()}

    def to_file(self, path: Path) -> None:
        with open(path, "w+") as f:
            json.dump(self._to_json(), f)

    def add(self, realization: int, entries: Set[Tuple[str, str, bool]]) -> None:
        if realization not in self._states:
            self._states[realization] = set()

        real_state = self._states[realization]
        assert real_state is not None

        for entry in entries:
            group, key, value = entry
            if (group, key, not value) in real_state:
                real_state.remove((group, key, not value))

        real_state.update(entries)

    def has(self, realization: int, key: str) -> bool:
        if realization not in self._states:
            return False

        state = self._states[realization]

        return (
            (key, key, True) in state
            or any(_has_it and key == _group for _group, _, _has_it in state)
            or any(_has_it and key == _key for _, _key, _has_it in state)
        )

    def has_entry(self, realization: int, key: str) -> bool:
        if realization not in self._states:
            return False

        state = self._states[realization]

        result = (
            (key, key, True) in state
            or (key, key, False) in state
            or any(key == _group for _group, _, _ in state)
            or any(key == _key for _, _key, _ in state)
        )

        return result


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

        self._realization_states = (
            RealizationState.from_file(self._path / "state_map.json")
            if os.path.exists(self._path / "state_map.json")
            else RealizationState()
        )

        self.__response_states_need_update = False  # Tmp
        self.__parameter_states_need_update = False  # Tmp
        self._has_invoked_refresh_statemap = False

    @property
    def _response_states_need_update(self) -> bool:
        return self.__response_states_need_update

    @_response_states_need_update.setter
    def _response_states_need_update(self, val: bool):
        if val and self._has_invoked_refresh_statemap:
            # Temp, all tests should pass without
            # hitting this line
            pass  # raise AssertionError("Expected this line to never be hit")

        self.__response_states_need_update = val

    @property
    def _parameter_states_need_update(self) -> bool:
        return self.__parameter_states_need_update

    @_parameter_states_need_update.setter
    def _parameter_states_need_update(self, val: bool):
        if val and self._has_invoked_refresh_statemap:
            # Temp, all tests should pass without
            # hitting this line
            pass

        self.__parameter_states_need_update = val

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

        with open(path / "index.json", mode="w", encoding="utf-8") as f:
            print(index.model_dump_json(), file=f)

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

    def get_realization_mask_without_parent_failure(self) -> npt.NDArray[np.bool_]:
        """
        Mask array indicating realizations without a parent failure.

        Returns
        -------
        parent_failures : ndarray of bool
            Boolean array where True means no parent failure.
        """

        return np.array(
            [
                (e != RealizationStorageState.PARENT_FAILURE)
                for e in self.get_ensemble_state()
            ]
        )

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
        Generates a mask array indicating which realizations have
        associated responses based on a given key.
        If no key is provided,
        it checks for any responses associated with each realization.

        Parameters
        ----------
        key : Optional[str]
            The specific response key to filter realizations.
            If `None`, all types of responses are considered.

        Returns
        -------
        NDArray[np.bool_]
            A boolean numpy array where each element is `True`
            if the corresponding realization has associated responses,
            and `False` otherwise.
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

        self.refresh_parameters_state_if_needed()

        return all(
            self._realization_states.has(realization, parameter)
            for parameter in self.experiment.parameter_configuration
        )

    def has_combined_response_dataset(self, key: str) -> bool:
        ds_key = self._find_unified_dataset_for_response(key)
        return (self._path / f"{ds_key}.nc").exists()

    def has_combined_parameter_dataset(self, key: str) -> bool:
        return (self._path / f"{key}.nc").exists()

    def _load_combined_response_dataset(self, key: str) -> xr.Dataset:
        ds_key = self._find_unified_dataset_for_response(key)

        unified_ds = xr.open_dataset(self._path / f"{ds_key}.nc")

        if key != ds_key:
            return unified_ds.sel(name=key, drop=True)

        return unified_ds

    def _load_combined_parameter_dataset(self, key: str) -> xr.Dataset:
        unified_ds = xr.open_dataset(self._path / f"{key}.nc")

        return unified_ds

    def refresh_responses_state_if_needed(self) -> None:
        if self._response_states_need_update:
            raise AssertionError("Expected this line to never be hit")
            self._response_states_need_update = False
            self._refresh_all_responses_state_for_all_realizations()

            self._realization_states.to_file(self._path / "state_map.json")

    def refresh_parameters_state_if_needed(self) -> None:
        if self._parameter_states_need_update:
            raise AssertionError("Expected this line to never be hit")
            self._parameter_states_need_update = False
            self._refresh_all_parameters_state_for_all_realizations()
            assert self._realization_states is not None
            self._realization_states.to_file(self._path / "state_map.json")

    def refresh_statemap(self):
        self._refresh_all_responses_state_for_all_realizations()
        self._refresh_all_parameters_state_for_all_realizations()
        self._parameter_states_need_update = False
        self._response_states_need_update = False
        self._has_invoked_refresh_statemap = True
        self._realization_states.to_file(self._path / "state_map.json")

    def _responses_exist_for_realization(
        self, realization: int, key: Optional[str] = None
    ) -> bool:
        """
        Determines whether responses exist for a specified realization
        in the ensemble, potentially filtered by a response key.

        Parameters
        ----------
        realization : int
            The index of the realization within the ensemble to check.
        key : str, optional
            Response key to filter realizations.
            If None, all responses are considered.

        Returns
        -------
        bool
            `True` if the specified responses exist for the given realization;
            otherwise, `False`.
        """

        self.refresh_responses_state_if_needed()
        assert self._realization_states is not None

        if not self.experiment.response_configuration:
            return True

        if key is not None:
            return self._realization_states.has(realization, key)

        return all(
            self._realization_states.has(realization, response_key)
            for response_key in self.experiment.response_configuration
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
                (self._realization_dir(i) / f"{parameter.name}.nc").exists()
                for parameter in self.experiment.parameter_configuration.values()
                if not parameter.forward_init
            )
            or all(
                (self._path / f"{parameter.name}.nc").exists()
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
        return [
            i
            for i in range(self.ensemble_size)
            if all(
                self._responses_exist_for_realization(i, response_key)
                for response_key in self.experiment.response_configuration
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

    def get_realization_with_responses(
        self, key: Optional[str] = None
    ) -> npt.NDArray[np.int_]:
        """
        Get an array of indices for realizations that have associated responses.

        Parameters
        ----------
        key : Optional[str], default None
            Response key to filter realizations. If None, all responses are considered.

        Returns
        -------
        npt.NDArray[np.int_]
            Array of realization indices with associated responses.
        """

        mask = self.get_realization_mask_with_responses(key)
        return np.flatnonzero(mask)

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
        with open(filename, mode="w", encoding="utf-8") as f:
            print(error.model_dump_json(), file=f)

    def unset_failure(
        self,
        realization: int,
    ) -> None:
        filename: Path = self._realization_dir(realization) / self._error_log_name
        if filename.exists():
            filename.unlink()

        for response_key in self.experiment.response_configuration:
            self._realization_states.clear_entry(realization, response_key)

        for parameter_group_key in self.experiment.parameter_configuration:
            self._realization_states.clear_entry(realization, parameter_group_key)

        self._refresh_all_responses_state_for_realization(realization)
        self._refresh_all_parameters_state_for_realization(realization)

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

        paths_to_check = [*self._path.glob("realization-*/summary.nc")]

        if os.path.exists(self._path / "summary.nc"):
            paths_to_check.append(self._path / "summary.nc")

        for p in paths_to_check:
            return sorted(xr.open_dataset(p)["name"].values)

        return []

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

    def load_parameters(
        self,
        group: str,
        realizations: Union[int, Iterable[int], None] = None,
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

        drop_reals_dim = isinstance(realizations, int)
        selected_realizations: Union[None, int, List[int]]
        if realizations is None:
            selected_realizations = None
        elif isinstance(realizations, int):
            assert isinstance(realizations, int)
            drop_reals_dim = True
            selected_realizations = realizations
        elif isinstance(realizations, (np.ndarray, tuple, list)):
            selected_realizations = list(realizations)
        else:
            raise ValueError(f"Invalid type for realizations: {type(realizations)}")

        try:
            ds = self.open_unified_parameter_dataset(group)
            if selected_realizations is None:
                return ds

            return ds.sel(realizations=selected_realizations, drop=drop_reals_dim)

        except (ValueError, KeyError, FileNotFoundError):
            # Fallback to check for real folder
            try:
                if isinstance(selected_realizations, int) and drop_reals_dim:
                    return xr.open_dataset(
                        self._path
                        / f"realization-{selected_realizations}"
                        / f"{group}.nc"
                    ).squeeze("realizations", drop=True)
                if selected_realizations is None:
                    return xr.combine_nested(
                        [
                            xr.open_dataset(p)
                            for p in self._path.glob(
                                f"realization-*/{glob.escape(group)}.nc"
                            )
                        ],
                        concat_dim="realizations",
                    )
                elif isinstance(selected_realizations, int):
                    return xr.open_dataset(
                        self._path
                        / f"realization-{selected_realizations}"
                        / f"{group}.nc"
                    )
                else:
                    assert isinstance(selected_realizations, list)
                    return xr.combine_nested(
                        [
                            xr.open_dataset(
                                self._path / f"realization-{real}" / f"{group}.nc"
                            )
                            for real in selected_realizations
                        ],
                        concat_dim="realizations",
                    )
            except FileNotFoundError as e:
                raise KeyError(
                    f"No dataset '{group}' in storage for "
                    f"realization {selected_realizations}"
                ) from e

    def _find_unified_dataset_for_response(self, key: str) -> str:
        all_gen_data_keys = {
            k
            for k, c in self.experiment.response_configuration.items()
            if isinstance(c, GenDataConfig)
        }

        if key == ResponseTypes.gen_data or key in all_gen_data_keys:
            return "gen_data"

        if key == ResponseTypes.summary or key in self.get_summary_keyset():
            return "summary"

        if key not in self.experiment.response_configuration:
            raise ValueError(f"{key} is not a response")

        return key

    def open_unified_response_dataset(self, key: str) -> xr.Dataset:
        dataset_key = self._find_unified_dataset_for_response(key)
        nc_path = self._path / f"{dataset_key}.nc"

        ds = None
        if os.path.exists(nc_path):
            ds = xr.open_dataset(nc_path)

        if not ds:
            raise FileNotFoundError(
                f"Dataset file for "
                f"{'response type' if key == dataset_key else 'response'} "
                f"{key} not found (tried {key}.nc)"
            )

        if key != dataset_key:
            return ds.sel(name=key, drop=True)

        return ds

    def open_unified_parameter_dataset(self, key: str) -> xr.Dataset:
        nc_path = self._path / f"{key}.nc"

        ds = None
        if os.path.exists(nc_path):
            ds = xr.open_dataset(nc_path)

        if not ds:
            raise FileNotFoundError(
                f"Dataset file for group {key} not found (tried {key}.nc)"
            )

        return ds

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
        dataset.to_netcdf(path=file_path, engine="scipy")

    def load_responses(
        self, key: str, realizations: Union[Tuple[int, ...], None] = None
    ) -> xr.Dataset:
        """Load responses for key and realizations into xarray Dataset.

        For each given realization, response data is loaded from the
        file whose filename matches the given key parameter.

        Parameters
        ----------
        key : str
            Response key to load.
        realizations : tuple of int
            Realization indices to load.

        Returns
        -------
        responses : Dataset
            Loaded xarray Dataset with responses.
        """

        try:
            ds = self.open_unified_response_dataset(key)
            if realizations:
                try:
                    return ds.sel(realization=list(realizations))
                except KeyError as err:
                    raise KeyError(
                        f"No response for key {key}, realization: {realizations}"
                    ) from err

            return ds
        except FileNotFoundError:
            # If the unified dataset does not exist,
            # we fall back to checking within the individual realization folders.
            if key == "gen_data":
                gen_data_keys = {
                    k
                    for k, c in self.experiment.response_configuration.items()
                    if isinstance(c, GenDataConfig)
                }
                return xr.concat(
                    [
                        self.load_responses(k, realizations).expand_dims(name=[k])
                        for k in gen_data_keys
                    ],
                    dim="name",
                ).transpose("realization", "name", "index", "report_step")

            datasets = [
                xr.open_dataset(self._path / f"realization-{real}" / f"{key}.nc")
                for real in (
                    realizations
                    if realizations is not None
                    else self.get_realization_with_responses(key).tolist()
                )
            ]

            if len(datasets) == 0:
                raise KeyError(
                    f"No response for key {key}, realization: {realizations}"
                ) from None

            return xr.combine_nested(datasets, concat_dim="realization")

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

        try:
            ds = self.load_responses(ResponseTypes.summary)
            if realization_index is not None:
                ds = ds.sel(realization=realization_index)

            df = ds.to_dataframe().pivot_table(
                index=["realization", "time"], columns="name", values="values"
            )
            df.index = df.index.rename(
                {"time": "Date", "realization": "Realization"}
            ).reorder_levels(["Realization", "Date"])
            df.axes[1].rename("", inplace=True)
            if keys:
                summary_keys = self.get_summary_keyset()
                summary_keys = sorted(
                    [key for key in keys if key in summary_keys]
                )  # ignore keys that doesn't exist
                return df[summary_keys]
            return df
        except (ValueError, KeyError, FileNotFoundError):
            return pd.DataFrame()

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

        dataframes: List[DataFrame] = []
        gen_kws = [
            config
            for config in self.experiment.parameter_configuration.values()
            if isinstance(config, GenKwConfig)
        ]
        if group:
            gen_kws = [config for config in gen_kws if config.name == group]
        for key in gen_kws:
            with contextlib.suppress(KeyError):
                ds = self.load_parameters(key.name)

                if realization_index is not None:
                    ds = ds.sel(realizations=realization_index)

                da = ds["transformed_values"]
                assert isinstance(da, xr.DataArray)
                da["names"] = np.char.add(f"{key.name}:", da["names"].astype(np.str_))
                df = da.to_dataframe().pivot_table(
                    index="realizations", columns="names", values="transformed_values"
                )
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

    def _validate_parameters_dataset(self, group: str, dataset: xr.Dataset) -> None:
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

    def _assert_dataset_not_already_created(
        self, group: str, realization: Optional[int] = None
    ) -> None:
        path_in_base_folder = self._path / f"{group}.nc"
        if os.path.exists(path_in_base_folder):
            f"There already exists a combined dataset for parameter group {group}"
            f" for group {group} @ {path_in_base_folder}. Parameters should be saved only once."

        if realization is not None:
            path_in_real_folder = self._realization_dir(realization) / f"{group}.nc"
            if os.path.exists(path_in_real_folder):
                raise KeyError(
                    "Detected attempt at overwriting already saved parameter"
                    f" for group {group} @ {path_in_real_folder}. Parameters should be saved only once."
                )

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

        self._assert_dataset_not_already_created(group)
        self._validate_parameters_dataset(group, dataset)

        path = self._realization_dir(realization) / f"{group}.nc"
        path.parent.mkdir(exist_ok=True)

        if "realizations" not in dataset.dims:
            dataset = dataset.expand_dims(realizations=[realization])

        dataset.to_netcdf(path, engine="scipy")

        self._realization_states.clear_entry(realization, group)
        self._parameter_states_need_update = True

    @require_write
    def save_response(self, group: str, data: xr.Dataset, realization: int) -> None:
        """
        Save dataset as response under group and realization index.

        Parameters
        ----------
        group : str
            Response group name for saving dataset.
        realization : int
            Realization index for saving group.
        data : Dataset
            Dataset to save.
        """

        if "values" not in data.variables:
            raise ValueError(
                f"Dataset for response group '{group}' "
                f"must contain a 'values' variable"
            )

        if data["values"].size == 0:
            raise ValueError(
                f"Responses {group} are empty. Cannot proceed with saving to storage."
            )

        if "realization" not in data.dims:
            data = data.expand_dims({"realization": [realization]})

        output_path = self._realization_dir(realization)
        Path.mkdir(output_path, parents=True, exist_ok=True)

        data.to_netcdf(output_path / f"{group}.nc", engine="scipy")
        self._response_states_need_update = True
        self._realization_states.clear_entry(realization, group)

    def _refresh_all_parameters_state_for_all_realizations(self) -> None:
        for real in range(self.ensemble_size):
            self._refresh_all_parameters_state_for_realization(realization=real)

        assert self._realization_states is not None
        self._realization_states.to_file(self._path / "state_map.json")

    def _refresh_all_responses_state_for_all_realizations(self) -> None:
        for real in range(self.ensemble_size):
            self._refresh_all_responses_state_for_realization(realization=real)

        self._realization_states.to_file(self._path / "state_map.json")

    def _refresh_all_responses_state_for_realization(self, realization: int) -> None:
        for response_key in self.experiment.response_configuration:
            self._refresh_response_state(response_key, realization)

    def _refresh_all_parameters_state_for_realization(self, realization: int) -> None:
        for parameter_key in self.experiment.parameter_configuration:
            self._refresh_parameter_state(parameter_key, realization)

    def _refresh_parameter_state(self, parameter_key: str, realization: int) -> None:
        if self._realization_states.has_entry(realization, parameter_key):
            return

        self._realization_states.add(
            realization,
            {
                (
                    parameter_key,
                    parameter_key,
                    os.path.exists(
                        self._realization_dir(realization) / f"{parameter_key}.nc"
                    ),
                )
            },
        )

    def _refresh_response_state(self, response_key: str, realization: int) -> None:
        if self._realization_states.has_entry(realization, response_key):
            return

        combined_ds_key = self._find_unified_dataset_for_response(response_key)

        # We assume we will never receive "sub-keys" for grouped datasets
        if combined_ds_key == "summary" and response_key != combined_ds_key:
            raise KeyError("Did not expect sub-key for grouped dataset")

        # ex: combined_ds_key == gen_data, response_key = WOPR_OP1
        # ex2: response_key = summary, combined_ds_key = summary
        is_grouped_ds = combined_ds_key == response_key
        has_realization_dir = os.path.exists(self._realization_dir(realization))

        if not has_realization_dir:
            self._realization_states.add(
                realization,
                {
                    (
                        combined_ds_key,
                        combined_ds_key if is_grouped_ds else response_key,
                        False,
                    )
                },
            )
            return

        if is_grouped_ds and os.path.exists(
            self._realization_dir(realization) / f"{combined_ds_key}.nc"
        ):
            self._realization_states.add(
                realization, {(combined_ds_key, combined_ds_key, True)}
            )
        else:
            self._realization_states.add(
                realization,
                {
                    (
                        combined_ds_key,
                        response_key,
                        os.path.exists(
                            self._realization_dir(realization) / f"{response_key}.nc"
                        ),
                    )
                },
            )

    def calculate_std_dev_for_parameter(self, parameter_group: str) -> xr.Dataset:
        if not parameter_group in self.experiment.parameter_configuration:
            raise ValueError(f"{parameter_group} is not registered to the experiment.")

        path_unified = self._path / f"{parameter_group}.nc"
        if os.path.exists(path_unified):
            return xr.open_dataset(path_unified).std("realizations")

        path = self._path / "realization-*" / f"{parameter_group}.nc"
        try:
            ds = xr.open_mfdataset(str(path))
        except OSError as e:
            raise e

        return ds.std("realizations")

    def get_observations_and_responses(
        self,
        observation_keys: List[str],
        active_realizations: Optional[npt.NDArray[np.int_]] = None,
    ) -> ObservationsAndResponsesData:
        """Return data grouped by observation name, showing the
        observation + std values, and accompanying simulated values per realization.

        * key_index is the "{time}" for summary, "{index},{report_step}" for gen_obs
        * Numbers 0...N correspond to the realization index

        Example output
                                  OBS  STD          0  ...        48         49
        name     key_index                             ...
        POLY_OBS [0, 0]      2.145705  0.6   3.721637  ...  0.862469   2.625992
                 [2, 0]      8.769220  1.4   6.419814  ...  1.304883   4.650068
                 [4, 0]     12.388015  3.0  12.796416  ...  2.535165   8.349348
                 [6, 0]     25.600465  5.4  22.851445  ...  4.553314  13.723831
                 [8, 0]     42.352048  8.6  36.584901  ...  7.359332  20.773518

        Arguments:
            observation_keys: List of observation names to include in the dataset
            active_realizations: List of active realization indices
        """

        numerical_data = []
        index_data = []

        reals_with_responses_mask = self.get_realization_with_responses()
        if active_realizations is not None:
            reals_with_responses_mask = np.intersect1d(
                active_realizations, reals_with_responses_mask
            )

        for response_type in self.experiment.observations:
            obs_datasets = self.experiment.observations[response_type]
            obs_names_to_check = set(obs_datasets["obs_name"].data).intersection(
                observation_keys
            )
            responses_ds = self.load_responses(
                response_type,
                realizations=tuple(reals_with_responses_mask),
            )

            index = ObservationsIndices[ResponseTypes(response_type)]
            for obs_name in obs_names_to_check:
                obs_ds = obs_datasets.sel(obs_name=obs_name, drop=True)

                obs_ds = obs_ds.dropna("name", subset=["observations"], how="all")
                for k in index:
                    obs_ds = obs_ds.dropna(dim=k, how="all")

                response_names_to_check = obs_ds["name"].data

                for response_name in response_names_to_check:
                    observations_for_response = obs_ds.sel(
                        name=response_name, drop=True
                    )

                    responses_matching_obs = responses_ds.sel(
                        name=response_name, drop=True
                    )

                    combined = observations_for_response.merge(
                        responses_matching_obs, join="left"
                    )

                    response_vals_per_real = (
                        combined["values"].stack(key=index).values.T
                    )

                    key_index_1d = np.array(
                        [
                            (
                                x.strftime("%Y-%m-%d")
                                if isinstance(x, pd.Timestamp)
                                else json.dumps(x)
                            )
                            for x in combined[index].coords.to_index()
                        ]
                    )
                    obs_vals_1d = combined["observations"].data
                    std_vals_1d = combined["std"].data

                    num_obs = len(obs_vals_1d)
                    obs_names_1d = np.array([obs_name] * num_obs)

                    if (
                        len(key_index_1d) != num_obs
                        or response_vals_per_real.shape[0] != num_obs
                        or len(std_vals_1d) != num_obs
                    ):
                        raise IndexError(
                            "Axis 0 misalignment, expected axis 0 length to "
                            f"correspond to observation names {num_obs}. Got:\n"
                            f"len(response_vals_per_real)={len(response_vals_per_real)}\n"
                            f"len(obs_names_1d)={len(obs_names_1d)}\n"
                            f"len(std_vals_1d)={len(std_vals_1d)}"
                        )

                    if response_vals_per_real.shape[1] != len(
                        reals_with_responses_mask
                    ):
                        raise IndexError(
                            "Axis 1 misalignment, expected axis 1 of"
                            f" response_vals_per_real to be the same as number of realizations"
                            f" with responses ({len(reals_with_responses_mask)}),"
                            f"but got response_vals_per_real.shape[1]"
                            f"={response_vals_per_real.shape[1]}"
                        )

                    _index_data = np.concatenate(
                        [
                            obs_names_1d.reshape(-1, 1),
                            key_index_1d.reshape(-1, 1),
                        ],
                        axis=1,
                    )
                    index_data.append(_index_data)
                    _numerical_data = np.concatenate(
                        [
                            obs_vals_1d.reshape(-1, 1),
                            std_vals_1d.reshape(-1, 1),
                            response_vals_per_real,
                        ],
                        axis=1,
                    )
                    numerical_data.append(_numerical_data)

        if not index_data:
            msg = (
                "No observation: "
                + (", ".join(observation_keys) if observation_keys is not None else "*")
                + " in ensemble"
            )
            raise KeyError(msg)

        index_df = pd.DataFrame(
            np.concatenate(index_data), columns=["name", "key_index"]
        )
        numerical_df = pd.DataFrame(
            np.concatenate(numerical_data),
            columns=["OBS", "STD"] + list(range(response_vals_per_real.shape[1])),
        )
        result_df = pd.concat([index_df, numerical_df], axis=1)
        result_df.sort_values(by=["name", "key_index"], inplace=True)
        return ObservationsAndResponsesData(result_df)

    def _unify_datasets(
        self,
        groups: List[str],
        concat_dim: Literal["realization", "realizations"],
        delete_after: bool = True,
    ) -> None:
        for group in groups:
            combined_ds_path = self._path / f"{group}.nc"
            has_existing_combined = os.path.exists(combined_ds_path)

            paths = sorted(
                self.mount_point.glob(f"realization-*/{glob.escape(group)}.nc")
            )

            if len(paths) > 0:
                new_combined = xr.combine_nested(
                    [xr.open_dataset(p, engine="scipy") for p in paths],
                    concat_dim=concat_dim,
                )

                if has_existing_combined:
                    # Merge new combined into old
                    old_combined = xr.open_dataset(combined_ds_path)
                    reals_to_replace = new_combined[concat_dim].data
                    reals_to_drop_from_old = set(reals_to_replace).intersection(
                        set(old_combined[concat_dim].data)
                    )

                    if reals_to_drop_from_old:
                        old_combined = old_combined.drop_sel(
                            {concat_dim: list(reals_to_drop_from_old)}
                        )

                    new_combined = old_combined.merge(new_combined)
                    os.remove(combined_ds_path)

                new_combined = ensure_correct_coordinate_order(new_combined)

                if not new_combined:
                    raise ValueError("Unified dataset somehow ended up empty")

                new_combined.to_netcdf(combined_ds_path, engine="scipy")

                if delete_after:
                    for p in paths:
                        os.remove(p)

    def unify_responses(self, key: Optional[str] = None) -> None:
        if key is None:
            for response_key in self.experiment.response_configuration:
                self.unify_responses(response_key)
                key = response_key

        gen_data_keys = {
            k
            for k, c in self.experiment.response_configuration.items()
            if isinstance(c, GenDataConfig)
        }

        if key == ResponseTypes.gen_data or key in gen_data_keys:
            has_existing_combined = os.path.exists(self._path / "gen_data.nc")

            # If gen data, combine across reals,
            # but also across all name(s) into one gen_data.nc

            files_to_remove = []
            to_concat = []
            for group in gen_data_keys:
                paths = sorted(
                    self.mount_point.glob(f"realization-*/{glob.escape(group)}.nc")
                )

                if len(paths) > 0:
                    ds_for_group = xr.concat(
                        [
                            ds.expand_dims(name=[group], axis=1)
                            for ds in [
                                xr.open_dataset(p, engine="scipy") for p in paths
                            ]
                        ],
                        dim="realization",
                    )
                    to_concat.append(ds_for_group)

                    files_to_remove.extend(paths)

            # Ensure deterministic ordering wrt name and real
            if to_concat:
                new_combined_ds = xr.concat(to_concat, dim="name").sortby(
                    ["realization", "name"]
                )
                new_combined_ds = ensure_correct_coordinate_order(new_combined_ds)

                if has_existing_combined:
                    old_combined = xr.load_dataset(self._path / "gen_data.nc")
                    updated_realizations = new_combined_ds["realization"].data
                    updated_realizations_in_old_combined = set(
                        updated_realizations
                    ).intersection(set(old_combined["realization"].data))

                    if updated_realizations_in_old_combined:
                        old_combined = old_combined.drop_sel(
                            {"realization": list(updated_realizations_in_old_combined)}
                        )

                    new_combined_ds = old_combined.merge(new_combined_ds)
                    os.remove(self._path / "gen_data.nc")

                new_combined_ds.to_netcdf(self._path / "gen_data.nc", engine="scipy")
                for f in files_to_remove:
                    os.remove(f)

        else:
            # If it is a summary, just combined across reals
            self._unify_datasets(
                (
                    [key]
                    if key is not None
                    else list(self.experiment.response_configuration.keys())
                ),
                "realization",
            )

    def unify_parameters(self, key: Optional[str] = None) -> None:
        self._unify_datasets(
            (
                [key]
                if key is not None
                else list(self.experiment.parameter_configuration.keys())
            ),
            "realizations",
        )

    def get_parameter_state(
        self, realization: int
    ) -> Dict[str, RealizationStorageState]:
        return {
            parameter: (
                RealizationStorageState.INITIALIZED
                if self._parameters_exist_for_realization(realization)
                else RealizationStorageState.UNDEFINED
            )
            for parameter in self.experiment.parameter_configuration
        }

    def get_response_state(
        self, realization: int
    ) -> Dict[str, RealizationStorageState]:
        return {
            response_key: (
                RealizationStorageState.HAS_DATA
                if self._responses_exist_for_realization(realization)
                else RealizationStorageState.UNDEFINED
            )
            for response_key in self.experiment.response_configuration
        }
