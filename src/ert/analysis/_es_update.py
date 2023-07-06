from __future__ import annotations

import logging
import time
from collections import UserDict
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import iterative_ensemble_smoother as ies
import numpy as np
import xarray as xr
from iterative_ensemble_smoother.experimental import (
    ensemble_smoother_update_step_row_scaling,
)

from ert._c_wrappers import update
from ert._c_wrappers.enkf.config.field_config import Field
from ert._c_wrappers.enkf.config.gen_kw_config import GenKwConfig
from ert._c_wrappers.enkf.config.surface_config import SurfaceConfig
from ert._c_wrappers.enkf.enums import RealizationStateEnum
from ert._c_wrappers.enkf.row_scaling import RowScaling

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert._c_wrappers.analysis import AnalysisModule
    from ert._c_wrappers.analysis.configuration import UpdateConfiguration
    from ert._c_wrappers.enkf import EnKFMain, EnkfObs
    from ert._c_wrappers.enkf.analysis_config import AnalysisConfig
    from ert._c_wrappers.enkf.ensemble_config import EnsembleConfig
    from ert.storage import EnsembleAccessor, EnsembleReader

_logger = logging.getLogger(__name__)


class ErtAnalysisError(Exception):
    pass


@dataclass
class UpdateSnapshot:
    obs_name: npt.NDArray[np.str_]
    obs_value: npt.NDArray[np.float32]
    obs_std: npt.NDArray[np.float32]
    obs_mask: npt.NDArray[np.bool_]
    response_mean: npt.NDArray[np.float32]
    response_std: npt.NDArray[np.float32]

    @property
    def obs_status(self) -> List[str]:
        return ["ACTIVE" if v else "DEACTIVATED" for v in self.obs_mask]


@dataclass
class SmootherSnapshot:
    source_case: str
    target_case: str
    analysis_module: str
    analysis_configuration: Dict[str, Any]
    alpha: float
    std_cutoff: float
    update_step_snapshots: Dict[str, "UpdateSnapshot"] = field(default_factory=dict)


@dataclass
class Progress:
    task: Task
    sub_task: Optional[Task]

    def __str__(self) -> str:
        ret = f"tasks: {self.task}"
        if self.sub_task is not None:
            ret += f"\nsub tasks: {self.sub_task}"
        return ret


@dataclass
class Task:
    description: str
    current: int
    total: Optional[int]

    def __str__(self) -> str:
        ret: str = f"running #{self.current}"
        if self.total is not None:
            ret += f" of {self.total}"
        return ret + f" - {self.description}"


ProgressCallback = Callable[[Progress], None]


def noop_progress_callback(_: Progress) -> None:
    pass


class TempStorage(UserDict):  # type: ignore
    def __getitem__(self, key: str) -> npt.NDArray[np.double]:
        value: Union[npt.NDArray[np.double], xr.DataArray] = self.data[key]
        if not isinstance(value, xr.DataArray):
            return value
        ensemble_size = len(value.realizations)
        return value.values.reshape(ensemble_size, -1).T

    def __setitem__(
        self, key: str, value: Union[npt.NDArray[np.double], xr.DataArray]
    ) -> None:
        old_value = self.data.get(key, None)
        if isinstance(old_value, xr.DataArray):
            old_value.data = value.T.reshape(*old_value.shape)
            self.data[key] = old_value
        else:
            self.data[key] = value

    def get_xr_array(self, key: str, real: int) -> xr.DataArray:
        value = self.data[key]
        if isinstance(value, xr.DataArray):
            return value[real]
        else:
            raise ValueError(f"TempStorage has no xarray DataFrame with key={key}")


def _get_A_matrix(
    temp_storage: TempStorage,
    parameters: List[update.Parameter],
) -> Optional["npt.NDArray[np.double]"]:
    matrices: List["npt.NDArray[np.double]"] = []
    for p in parameters:
        if p.index_list is None:
            matrices.append(temp_storage[p.name])
        else:
            matrices.append(temp_storage[p.name][p.index_list, :])
    return np.vstack(matrices) if matrices else None


def _param_ensemble_for_projection(
    temp_storage: TempStorage,
    ensemble_size: int,
    updatestep: UpdateConfiguration,
) -> Optional[npt.NDArray[np.double]]:
    """Responses must be projected when num_params < ensemble_size - 1.
    The number of parameters here refers to the total number of parameters used to
    drive the model and not the number of parameters in a local update step.

    Scenario 1:
        - p < N - 1, meaning that projection needs to be done.
        - User wants to loop through each parameter and update it separately,
          perhaps to do distance based localization.
        In this case, we need to pass `param_ensemble` to the smoother
        so that the responses are projected using the same `param_ensemble`
        in every loop.
    Scenario 2:
        - p > N - 1, meaning that projection is not necessary.
        - User wants to loop through every parameter as in Scenario 1.
        In this case `param_ensemble` should be `None` which means
        that no projection will be done even when updating a single parameter.
    """
    num_params = sum(arr.shape[0] for arr in temp_storage.values())
    if num_params < ensemble_size - 1:
        _params: List[List[update.Parameter]] = [
            update_step.parameters for update_step in updatestep
        ]
        # Flatten list of lists
        params: List[update.Parameter] = [
            item for sublist in _params for item in sublist
        ]
        return _get_A_matrix(temp_storage, params)
    return None


def _get_params_with_row_scaling(
    temp_storage: TempStorage,
    parameters: List[update.RowScalingParameter],
) -> List[Tuple[npt.NDArray[np.double], RowScaling]]:
    matrices = []
    for p in parameters:
        if p.index_list is None:
            matrices.append((temp_storage[p.name], p.row_scaling))
        else:
            matrices.append(
                (
                    temp_storage[p.name][p.index_list, :],
                    p.row_scaling,
                ),
            )

    return matrices


def _save_to_temp_storage(
    temp_storage: TempStorage,
    parameters: List[update.Parameter],
    A: Optional["npt.NDArray[np.double]"],
) -> None:
    if A is None:
        return
    offset = 0
    for p in parameters:
        if p.index_list is None:
            rows = temp_storage[p.name].shape[0]
            temp_storage[p.name] = A[offset : offset + rows, :]
            offset += rows
        else:
            row_indices = p.index_list
            for i, row in enumerate(row_indices):
                temp_storage[p.name][row] = A[offset + i]
            offset += len(row_indices)


def _save_temp_storage_to_disk(
    target_fs: EnsembleAccessor,
    ensemble_config: "EnsembleConfig",
    temp_storage: TempStorage,
    iens_active_index: List[int],
) -> None:
    for key, matrix in temp_storage.items():
        config_node = ensemble_config.parameter_configs[key]
        for i, realization in enumerate(iens_active_index):
            if isinstance(config_node, GenKwConfig):
                assert isinstance(matrix, np.ndarray)
                dataset = xr.Dataset(
                    {
                        "values": ("names", matrix[:, i]),
                        "transformed_values": (
                            "names",
                            config_node.transform(matrix[:, i]),
                        ),
                        "names": [e.name for e in config_node.transfer_functions],
                    }
                )
                target_fs.save_parameters(key, realization, dataset)
            elif isinstance(config_node, (Field, SurfaceConfig)):
                _matrix = temp_storage.get_xr_array(key, i)
                assert isinstance(_matrix, xr.DataArray)
                target_fs.save_parameters(key, realization, _matrix.to_dataset())
            else:
                raise NotImplementedError(f"{type(config_node)} is not supported")


def _create_temporary_parameter_storage(
    source_fs: EnsembleReader,
    ensemble_config: "EnsembleConfig",
    iens_active_index: List[int],
) -> TempStorage:
    temp_storage = TempStorage()
    t_genkw = 0.0
    t_surface = 0.0
    t_field = 0.0
    _logger.debug("_create_temporary_parameter_storage() - start")
    for key in ensemble_config.parameters:
        config_node = ensemble_config.parameter_configs[key]
        matrix: Union[npt.NDArray[np.double], xr.DataArray]
        if isinstance(config_node, GenKwConfig):
            t = time.perf_counter()
            matrix = source_fs.load_parameters(key, iens_active_index).values.T
            t_genkw += time.perf_counter() - t
        elif isinstance(config_node, SurfaceConfig):
            t = time.perf_counter()
            matrix = source_fs.load_parameters(key, iens_active_index)
            t_surface += time.perf_counter() - t
        elif isinstance(config_node, Field):
            t = time.perf_counter()
            matrix = source_fs.load_parameters(key, iens_active_index)
            t_field += time.perf_counter() - t
        else:
            raise NotImplementedError(f"{type(config_node)} is not supported")
        temp_storage[key] = matrix
        _logger.debug(
            f"_create_temporary_parameter_storage() time_used gen_kw={t_genkw:.4f}s, \
                  surface={t_surface:.4f}s, field={t_field:.4f}s"
        )
    return temp_storage


def _get_obs_and_measure_data(
    obs: EnkfObs,
    source_fs: EnsembleReader,
    selected_observations: List[Tuple[str, List[int]]],
    ens_active_list: Tuple[int, ...],
) -> Tuple[
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.str_],
]:
    ens_active_list = tuple(ens_active_list)
    measured_data = []
    observation_keys = []
    observation_values = []
    observation_errors = []
    for obs_key, obs_active_list in selected_observations:
        group, observation = obs.get_dataset(obs_key)
        if obs_active_list:
            index = observation.coords.to_index()[obs_active_list]
            sub_selection = {
                name: list(set(index.get_level_values(name))) for name in index.names
            }
            observation = observation.sel(sub_selection)
        ds = source_fs.load_response(group, ens_active_list)
        try:
            filtered_ds = observation.merge(ds, join="left")
        except KeyError:
            raise ErtAnalysisError(
                f"Mismatched index for: "
                f"Observation: {obs_key} attached to response: {group}"
            )

        observation_keys.append([obs_key] * len(filtered_ds.observations.data.ravel()))
        observation_values.append(filtered_ds["observations"].data.ravel())
        observation_errors.append(filtered_ds["std"].data.ravel())
        measured_data.append(
            filtered_ds["values"]
            .transpose(..., "realization")
            .values.reshape((-1, len(filtered_ds.realization)))
        )
    source_fs.load_response.cache_clear()
    return (
        np.concatenate(measured_data, axis=0),
        np.concatenate(observation_values),
        np.concatenate(observation_errors),
        np.concatenate(observation_keys),
    )


def _load_observations_and_responses(
    source_fs: EnsembleReader,
    obs: EnkfObs,
    alpha: float,
    std_cutoff: float,
    global_std_scaling: float,
    ens_mask: List[bool],
    selected_observations: List[Tuple[str, List[int]]],
) -> Any:
    ens_active_list = tuple(i for i, b in enumerate(ens_mask) if b)

    S, observations, errors, obs_keys = _get_obs_and_measure_data(
        obs,
        source_fs,
        selected_observations,
        ens_active_list,
    )

    # Inflating measurement errors by a factor sqrt(global_std_scaling) as shown
    # in for example evensen2018 - Analysis of iterative ensemble smoothers for
    # solving inverse problems.
    # `global_std_scaling` is 1.0 for ES.
    errors *= sqrt(global_std_scaling)

    ens_mean = S.mean(axis=1)
    ens_std = S.std(ddof=0, axis=1)

    ens_std_mask = ens_std > std_cutoff
    ens_mean_mask = abs(observations - ens_mean) <= alpha * (ens_std + errors)

    obs_mask = np.logical_and(ens_mean_mask, ens_std_mask)

    update_snapshot = UpdateSnapshot(
        obs_name=obs_keys,
        obs_value=observations,
        obs_std=errors,
        obs_mask=obs_mask,
        response_mean=ens_mean,
        response_std=ens_std,
    )
    for missing_obs in obs_keys[~obs_mask]:
        _logger.warning(f"Deactivating observation: {missing_obs}")

    return S[obs_mask], (
        observations[obs_mask],
        errors[obs_mask],
        update_snapshot,
    )


def analysis_ES(
    updatestep: "UpdateConfiguration",
    obs: EnkfObs,
    rng: np.random.Generator,
    module: "AnalysisModule",
    alpha: float,
    std_cutoff: float,
    global_scaling: float,
    smoother_snapshot: SmootherSnapshot,
    ens_mask: List[bool],
    ensemble_config: "EnsembleConfig",
    source_fs: EnsembleReader,
    target_fs: EnsembleAccessor,
    progress_callback: ProgressCallback,
) -> None:
    iens_active_index = [i for i in range(len(ens_mask)) if ens_mask[i]]

    progress_callback(Progress(Task("Loading data", 1, 3), None))
    temp_storage = _create_temporary_parameter_storage(
        source_fs, ensemble_config, iens_active_index
    )

    ensemble_size = sum(ens_mask)
    param_ensemble = _param_ensemble_for_projection(
        temp_storage, ensemble_size, updatestep
    )

    progress_callback(Progress(Task("Updating data", 2, 3), None))
    for update_step in updatestep:
        try:
            S, (
                observation_values,
                observation_errors,
                update_snapshot,
            ) = _load_observations_and_responses(
                source_fs,
                obs,
                alpha,
                std_cutoff,
                global_scaling,
                ens_mask,
                update_step.observation_config(),
            )
        except IndexError as e:
            raise ErtAnalysisError(e) from e

        # pylint: disable=unsupported-assignment-operation
        smoother_snapshot.update_step_snapshots[update_step.name] = update_snapshot
        if len(observation_values) == 0:
            raise ErtAnalysisError(
                f"No active observations for update step: {update_step.name}."
            )

        noise = rng.standard_normal(size=(len(observation_values), S.shape[1]))

        smoother = ies.ES()
        for parameter in update_step.parameters:
            smoother.fit(
                S,
                observation_errors,
                observation_values,
                noise=noise,
                truncation=module.get_truncation(),
                inversion=ies.InversionType(module.inversion),
                param_ensemble=param_ensemble,
            )
            if active_indices := parameter.index_list:
                temp_storage[parameter.name][active_indices, :] = smoother.update(
                    temp_storage[parameter.name][active_indices, :]
                )
            else:
                temp_storage[parameter.name] = smoother.update(
                    temp_storage[parameter.name]
                )

        if params_with_row_scaling := _get_params_with_row_scaling(
            temp_storage, update_step.row_scaling_parameters
        ):
            params_with_row_scaling = ensemble_smoother_update_step_row_scaling(
                S,
                params_with_row_scaling,
                observation_errors,
                observation_values,
                noise,
                module.get_truncation(),
                ies.InversionType(module.inversion),
            )
            for parameter, (A, _) in zip(
                update_step.row_scaling_parameters, params_with_row_scaling
            ):
                _save_to_temp_storage(temp_storage, [parameter], A)

    progress_callback(Progress(Task("Storing data", 3, 3), None))
    _save_temp_storage_to_disk(
        target_fs, ensemble_config, temp_storage, iens_active_index
    )


def analysis_IES(
    updatestep: "UpdateConfiguration",
    obs: EnkfObs,
    rng: np.random.Generator,
    module: "AnalysisModule",
    alpha: float,
    std_cutoff: float,
    global_scaling: float,
    smoother_snapshot: SmootherSnapshot,
    ens_mask: List[bool],
    ensemble_config: "EnsembleConfig",
    source_fs: EnsembleReader,
    target_fs: EnsembleAccessor,
    iterative_ensemble_smoother: ies.SIES,
    progress_callback: ProgressCallback,
) -> None:
    iens_active_index = [i for i in range(len(ens_mask)) if ens_mask[i]]

    progress_callback(Progress(Task("Loading data", 1, 3), None))
    temp_storage = _create_temporary_parameter_storage(
        source_fs, ensemble_config, iens_active_index
    )
    progress_callback(Progress(Task("Updating data", 2, 3), None))

    ensemble_size = sum(ens_mask)
    param_ensemble = _param_ensemble_for_projection(
        temp_storage, ensemble_size, updatestep
    )

    # Looping over local analysis update_step
    for update_step in updatestep:
        try:
            S, (
                observation_values,
                observation_errors,
                update_snapshot,
            ) = _load_observations_and_responses(
                source_fs,
                obs,
                alpha,
                std_cutoff,
                global_scaling,
                ens_mask,
                update_step.observation_config(),
            )
        except IndexError as e:
            raise ErtAnalysisError(e)
        # pylint: disable=unsupported-assignment-operation
        smoother_snapshot.update_step_snapshots[update_step.name] = update_snapshot
        if len(observation_values) == 0:
            raise ErtAnalysisError(
                f"No active observations for update step: {update_step.name}."
            )

        noise = rng.standard_normal(size=(len(observation_values), S.shape[1]))
        for parameter in update_step.parameters:
            iterative_ensemble_smoother.fit(
                S,
                observation_errors,
                observation_values,
                noise=noise,
                ensemble_mask=np.array(ens_mask),
                inversion=ies.InversionType(module.inversion),
                truncation=module.get_truncation(),
                param_ensemble=param_ensemble,
            )
            if active_indices := parameter.index_list:
                temp_storage[parameter.name][
                    active_indices, :
                ] = iterative_ensemble_smoother.update(
                    temp_storage[parameter.name][active_indices, :]
                )
            else:
                temp_storage[parameter.name] = iterative_ensemble_smoother.update(
                    temp_storage[parameter.name]
                )

    progress_callback(Progress(Task("Storing data", 3, 3), None))
    _save_temp_storage_to_disk(
        target_fs, ensemble_config, temp_storage, iens_active_index
    )


def _write_update_report(fname: Path, snapshot: SmootherSnapshot) -> None:
    # Make sure log file parents exist
    fname.parent.mkdir(parents=True, exist_ok=True)
    for update_step_name, update_step in snapshot.update_step_snapshots.items():
        with open(fname, "w", encoding="utf-8") as fout:
            fout.write("=" * 127 + "\n")
            fout.write("Report step...: deprecated\n")
            fout.write(f"Update step......: {update_step_name:<10}\n")
            fout.write("-" * 127 + "\n")
            fout.write(
                "Observed history".rjust(73)
                + "|".rjust(16)
                + "Simulated data".rjust(27)
                + "\n".rjust(9)
            )
            fout.write("-" * 127 + "\n")
            for nr, (name, val, std, status, ens_val, ens_std) in enumerate(
                zip(
                    update_step.obs_name,
                    update_step.obs_value,
                    update_step.obs_std,
                    update_step.obs_status,
                    update_step.response_mean,
                    update_step.response_std,
                )
            ):
                if status in ["DEACTIVATED", "LOCAL_INACTIVE"]:
                    status = "Inactive"
                fout.write(
                    f"{nr+1:^6}: {name:30} {val:>16.3f} +/- {std:>17.3f} "
                    f"{status.capitalize():9} | {ens_val:>17.3f} +/- {ens_std:>15.3f}  "
                    f"\n"
                )
            fout.write("=" * 127 + "\n")


def _assert_has_enough_realizations(
    ens_mask: List[bool], analysis_config: "AnalysisConfig"
) -> None:
    active_realizations = sum(ens_mask)
    if not analysis_config.have_enough_realisations(active_realizations):
        raise ErtAnalysisError(
            f"There are {active_realizations} active realisations left, which is "
            "less than the minimum specified - stopping assimilation.",
        )


def _create_smoother_snapshot(
    prior_name: "str", posterior_name: "str", analysis_config: "AnalysisConfig"
) -> SmootherSnapshot:
    return SmootherSnapshot(
        prior_name,
        posterior_name,
        analysis_config.active_module_name(),
        analysis_config.get_active_module().variable_value_dict(),
        analysis_config.get_enkf_alpha(),
        analysis_config.get_std_cutoff(),
    )


class ESUpdate:
    def __init__(self, enkf_main: "EnKFMain"):
        self.ert = enkf_main
        self.update_snapshots: Dict[str, SmootherSnapshot] = {}

    def smootherUpdate(
        self,
        prior_storage: EnsembleReader,
        posterior_storage: EnsembleAccessor,
        run_id: str,
        progress_callback: Optional[ProgressCallback] = None,
        global_scaling: float = 1.0,
    ) -> None:
        if not progress_callback:
            progress_callback = noop_progress_callback

        updatestep = self.ert.getLocalConfig()

        analysis_config = self.ert.analysisConfig()
        obs = self.ert.getObservations()
        ensemble_config = self.ert.ensembleConfig()

        alpha = analysis_config.get_enkf_alpha()
        std_cutoff = analysis_config.get_std_cutoff()
        ens_mask = prior_storage.get_realization_mask_from_state(
            [RealizationStateEnum.STATE_HAS_DATA]
        )
        _assert_has_enough_realizations(ens_mask, analysis_config)

        smoother_snapshot = _create_smoother_snapshot(
            prior_storage.name, posterior_storage.name, analysis_config
        )

        analysis_ES(
            updatestep,
            obs,
            self.ert.rng(),
            analysis_config.get_active_module(),
            alpha,
            std_cutoff,
            global_scaling,
            smoother_snapshot,
            ens_mask,
            ensemble_config,
            prior_storage,
            posterior_storage,
            progress_callback,
        )

        _write_update_report(
            Path(analysis_config.get_log_path()) / "deprecated", smoother_snapshot
        )

        self.update_snapshots[run_id] = smoother_snapshot

    def iterative_smoother_update(
        self,
        prior_storage: EnsembleReader,
        posterior_storage: EnsembleAccessor,
        w_container: ies.SIES,
        run_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        if not progress_callback:
            progress_callback = noop_progress_callback

        updatestep = self.ert.getLocalConfig()
        if len(updatestep) > 1:
            raise ErtAnalysisError(
                "Can not combine IES_ENKF modules with multi step updates"
            )

        analysis_config = self.ert.analysisConfig()

        obs = self.ert.getObservations()
        ensemble_config = self.ert.ensembleConfig()

        alpha = analysis_config.get_enkf_alpha()
        std_cutoff = analysis_config.get_std_cutoff()
        ens_mask = prior_storage.get_realization_mask_from_state(
            [RealizationStateEnum.STATE_HAS_DATA]
        )

        _assert_has_enough_realizations(ens_mask, analysis_config)

        smoother_snapshot = _create_smoother_snapshot(
            prior_storage.name, posterior_storage.name, analysis_config
        )

        analysis_IES(
            updatestep,
            obs,
            self.ert.rng(),
            analysis_config.get_active_module(),
            alpha,
            std_cutoff,
            1.0,
            smoother_snapshot,
            ens_mask,
            ensemble_config,
            prior_storage,
            posterior_storage,
            w_container,
            progress_callback,
        )

        _write_update_report(
            Path(analysis_config.get_log_path()) / "deprecated", smoother_snapshot
        )

        self.update_snapshots[run_id] = smoother_snapshot
