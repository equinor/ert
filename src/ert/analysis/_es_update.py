from __future__ import annotations

import logging
import time
from collections import UserDict, defaultdict
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    DefaultDict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import iterative_ensemble_smoother as ies
import numpy as np
import psutil
import xarray as xr
from iterative_ensemble_smoother.experimental import (
    AdaptiveESMDA,
)
from typing_extensions import Self

from ..config.analysis_config import ObservationGroups, UpdateSettings
from ..config.analysis_module import ESSettings, IESSettings
from . import misfit_preprocessor
from .event import (
    AnalysisErrorEvent,
    AnalysisEvent,
    AnalysisStatusEvent,
    AnalysisTimeEvent,
)
from .snapshots import (
    ObservationAndResponseSnapshot,
    ObservationStatus,
    SmootherSnapshot,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble, Experiment

_logger = logging.getLogger(__name__)


class ErtAnalysisError(Exception):
    pass


def noop_progress_callback(_: AnalysisEvent) -> None:
    pass


T = TypeVar("T")


class TimedIterator(Generic[T]):
    def __init__(
        self, iterable: Sequence[T], callback: Callable[[AnalysisEvent], None]
    ) -> None:
        self._start_time = time.perf_counter()
        self._iterable = iterable
        self._callback = callback
        self._index = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        try:
            result = self._iterable[self._index]
        except IndexError as e:
            raise StopIteration from e

        if self._index != 0:
            elapsed_time = time.perf_counter() - self._start_time
            estimated_remaining_time = (elapsed_time / (self._index)) * (
                len(self._iterable) - self._index
            )
            self._callback(
                AnalysisTimeEvent(
                    remaining_time=estimated_remaining_time, elapsed_time=elapsed_time
                )
            )

        self._index += 1
        return result


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
        old_value = self.data.get(key)
        if isinstance(old_value, xr.DataArray):
            old_value.data = value.T.reshape(*old_value.shape)
            self.data[key] = old_value
        else:
            self.data[key] = value


def _all_parameters(
    ensemble: Ensemble,
    iens_active_index: npt.NDArray[np.int_],
    param_groups: List[str],
) -> Optional[npt.NDArray[np.double]]:
    """Return all parameters in assimilation problem"""

    temp_storage = TempStorage()
    for param_group in param_groups:
        _temp_storage = _create_temporary_parameter_storage(
            ensemble, iens_active_index, param_group
        )
        temp_storage[param_group] = _temp_storage[param_group]
    matrices = [temp_storage[p] for p in param_groups]
    return np.vstack(matrices) if matrices else None


def _save_temp_storage_to_disk(
    ensemble: Ensemble,
    temp_storage: TempStorage,
    iens_active_index: npt.NDArray[np.int_],
) -> None:
    for key, matrix in temp_storage.items():
        config_node = ensemble.experiment.parameter_configuration[key]
        for i, realization in enumerate(iens_active_index):
            config_node.save_parameters(ensemble, key, realization, matrix[:, i])


def _create_temporary_parameter_storage(
    ensemble: Ensemble,
    iens_active_index: npt.NDArray[np.int_],
    param_group: str,
) -> TempStorage:
    temp_storage = TempStorage()
    config_node = ensemble.experiment.parameter_configuration[param_group]
    temp_storage[param_group] = config_node.load_parameters(
        ensemble, param_group, iens_active_index
    )
    return temp_storage


def _get_obs_and_measure_data(
    ensemble: Ensemble,
    selected_observations: Iterable[str],
    iens_active_index: npt.NDArray[np.int_],
) -> Tuple[
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.str_],
]:
    measured_data = []
    observation_keys = []
    observation_values = []
    observation_errors = []
    observations = ensemble.experiment.observations
    for obs in selected_observations:
        observation = observations[obs]
        group = observation.attrs["response"]
        response = ensemble.load_responses(group, tuple(iens_active_index))
        if "time" in observation.coords:
            response = response.reindex(
                time=observation.time, method="nearest", tolerance="1s"  # type: ignore
            )
        try:
            filtered_response = observation.merge(response, join="left")
        except KeyError as e:
            raise ErtAnalysisError(
                f"Mismatched index for: "
                f"Observation: {obs} attached to response: {group}"
            ) from e

        observation_keys.append([obs] * filtered_response["observations"].size)
        observation_values.append(filtered_response["observations"].data.ravel())
        observation_errors.append(filtered_response["std"].data.ravel())
        measured_data.append(
            filtered_response["values"]
            .transpose(..., "realization")
            .values.reshape((-1, len(filtered_response.realization)))
        )
    ensemble.load_responses.cache_clear()
    return (
        np.concatenate(measured_data, axis=0),
        np.concatenate(observation_values),
        np.concatenate(observation_errors),
        np.concatenate(observation_keys),
    )


def _expand_wildcards(
    input_list: npt.NDArray[np.str_], patterns: List[str]
) -> List[str]:
    matches = []
    for pattern in patterns:
        matches.extend([val for val in input_list if fnmatch(val, pattern)])
    return list(set(matches))


def _load_observations_and_responses(
    ensemble: Ensemble,
    alpha: float,
    std_cutoff: float,
    global_std_scaling: float,
    iens_active_index: npt.NDArray[np.int_],
    selected_observations: Iterable[str],
    auto_scale_observations: Optional[List[ObservationGroups]],
) -> Tuple[
    npt.NDArray[np.float_],
    Tuple[
        npt.NDArray[np.float_],
        npt.NDArray[np.float_],
        List[ObservationAndResponseSnapshot],
    ],
]:
    S, observations, errors, obs_keys = _get_obs_and_measure_data(
        ensemble,
        selected_observations,
        iens_active_index,
    )

    # Inflating measurement errors by a factor sqrt(global_std_scaling) as shown
    # in for example evensen2018 - Analysis of iterative ensemble smoothers for
    # solving inverse problems.
    # `global_std_scaling` is 1.0 for ES.
    scaling = np.sqrt(global_std_scaling) * np.ones_like(errors)
    scaled_errors = errors * scaling

    # Identifies non-outlier observations based on responses.
    ens_mean = S.mean(axis=1)
    ens_std = S.std(ddof=0, axis=1)
    ens_std_mask = ens_std > std_cutoff
    ens_mean_mask = abs(observations - ens_mean) <= alpha * (ens_std + scaled_errors)
    obs_mask = np.logical_and(ens_mean_mask, ens_std_mask)

    if auto_scale_observations:
        for input_group in auto_scale_observations:
            group = _expand_wildcards(obs_keys, input_group)
            obs_group_mask = np.isin(obs_keys, group) & obs_mask
            scaling[obs_group_mask] *= misfit_preprocessor.main(
                S[obs_group_mask], scaled_errors[obs_group_mask]
            )

    update_snapshot = []
    for (
        obs_name,
        obs_val,
        obs_std,
        obs_scaling,
        response_mean,
        response_std,
        response_mean_mask,
        response_std_mask,
    ) in zip(
        obs_keys,
        observations,
        errors,
        scaling,
        ens_mean,
        ens_std,
        ens_mean_mask,
        ens_std_mask,
    ):
        update_snapshot.append(
            ObservationAndResponseSnapshot(
                obs_name=obs_name,
                obs_val=obs_val,
                obs_std=obs_std,
                obs_scaling=obs_scaling,
                response_mean=response_mean,
                response_std=response_std,
                response_mean_mask=response_mean_mask,
                response_std_mask=response_std_mask,
            )
        )

    for missing_obs in obs_keys[~obs_mask]:
        _logger.warning(f"Deactivating observation: {missing_obs}")

    return S[obs_mask], (
        observations[obs_mask],
        scaled_errors[obs_mask],
        update_snapshot,
    )


def _split_by_batchsize(
    arr: npt.NDArray[np.int_], batch_size: int
) -> List[npt.NDArray[np.int_]]:
    """
    Splits an array into sub-arrays of a specified batch size.

    Examples
    --------
    >>> num_params = 10
    >>> batch_size = 3
    >>> s = np.arange(0, num_params)
    >>> _split_by_batchsize(s, batch_size)
    [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]

    >>> num_params = 10
    >>> batch_size = 10
    >>> s = np.arange(0, num_params)
    >>> _split_by_batchsize(s, batch_size)
    [array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]

    >>> num_params = 10
    >>> batch_size = 20
    >>> s = np.arange(0, num_params)
    >>> _split_by_batchsize(s, batch_size)
    [array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
    """
    sections = 1 if batch_size > len(arr) else len(arr) // batch_size
    return np.array_split(arr, sections)


def _calculate_adaptive_batch_size(num_params: int, num_obs: int) -> int:
    """Calculate adaptive batch size to optimize memory usage during Adaptive Localization
    Adaptive Localization calculates the cross-covariance between parameters and responses.
    Cross-covariance is a matrix with shape num_params x num_obs which may be larger than memory.
    Therefore, a batching algorithm is used where only a subset of parameters is used when
    calculating cross-covariance.
    This function calculates a batch size that can fit into the available memory, accounting
    for a safety margin.

    Derivation of formula:
    ---------------------
    available_memory = (amount of available memory on system) * memory_safety_factor
    required_memory = num_params * num_obs * bytes_in_float32
    num_params = required_memory / (num_obs * bytes_in_float32)
    We want (required_memory < available_memory) so:
    num_params < available_memory / (num_obs * bytes_in_float32)

    The available memory is checked using the `psutil` library, which provides information about
    system memory usage.
    From `psutil` documentation:
    - available:
        the memory that can be given instantly to processes without the
        system going into swap.
        This is calculated by summing different memory values depending
        on the platform and it is supposed to be used to monitor actual
        memory usage in a cross platform fashion.
    """
    available_memory_in_bytes = psutil.virtual_memory().available
    memory_safety_factor = 0.8
    # Fields are stored as 32-bit floats.
    bytes_in_float32 = 4
    return min(
        int(
            np.floor(
                (available_memory_in_bytes * memory_safety_factor)
                / (num_obs * bytes_in_float32)
            )
        ),
        num_params,
    )


def _copy_unupdated_parameters(
    all_parameter_groups: Iterable[str],
    updated_parameter_groups: Iterable[str],
    iens_active_index: npt.NDArray[np.int_],
    source_ensemble: Ensemble,
    target_ensemble: Ensemble,
) -> None:
    """
    Copies parameter groups that have not been updated from a source ensemble to a target ensemble.
    This function ensures that all realizations in the target ensemble have a complete set of parameters,
    including those that were not updated.
    This is necessary because users can choose not to update parameters but may still want to analyse them.

    Parameters:
    all_parameter_groups (List[str]): A list of all parameter groups.
    updated_parameter_groups (List[str]): A list of parameter groups that have already been updated.
    iens_active_index (npt.NDArray[np.int_]): An array of indices for the active realizations in the
                                              target ensemble.
    source_ensemble (Ensemble): The file system of the source ensemble, from which parameters are copied.
    target_ensemble (Ensemble): The file system of the target ensemble, to which parameters are saved.

    Returns:
    None: The function does not return any value but updates the target file system by copying over
    the parameters.
    """
    # Identify parameter groups that have not been updated
    not_updated_parameter_groups = list(
        set(all_parameter_groups) - set(updated_parameter_groups)
    )

    # Copy the non-updated parameter groups from source to target for each active realization
    for parameter_group in not_updated_parameter_groups:
        for realization in iens_active_index:
            ds = source_ensemble.load_parameters(parameter_group, int(realization))
            target_ensemble.save_parameters(parameter_group, realization, ds)


def analysis_ES(
    parameters: Iterable[str],
    observations: Iterable[str],
    rng: np.random.Generator,
    module: ESSettings,
    alpha: float,
    std_cutoff: float,
    global_scaling: float,
    smoother_snapshot: SmootherSnapshot,
    ens_mask: npt.NDArray[np.bool_],
    source_ensemble: Ensemble,
    target_ensemble: Ensemble,
    progress_callback: Callable[[AnalysisEvent], None],
    auto_scale_observations: Optional[List[ObservationGroups]],
) -> None:
    iens_active_index = np.flatnonzero(ens_mask)

    ensemble_size = ens_mask.sum()

    def adaptive_localization_progress_callback(
        iterable: Sequence[T],
    ) -> TimedIterator[T]:
        return TimedIterator(iterable, progress_callback)

    progress_callback(AnalysisStatusEvent(msg="Loading observations and responses.."))
    (
        S,
        (
            observation_values,
            observation_errors,
            update_snapshot,
        ),
    ) = _load_observations_and_responses(
        source_ensemble,
        alpha,
        std_cutoff,
        global_scaling,
        iens_active_index,
        observations,
        auto_scale_observations,
    )
    num_obs = len(observation_values)

    smoother_snapshot.update_step_snapshots = update_snapshot

    if num_obs == 0:
        msg = "No active observations for update step"
        progress_callback(
            AnalysisErrorEvent(error_msg=msg, smoother_snapshot=smoother_snapshot)
        )
        raise ErtAnalysisError(msg)

    smoother_es = ies.ESMDA(
        covariance=observation_errors**2,
        observations=observation_values,
        alpha=1,  # The user is responsible for scaling observation covariance (esmda usage)
        seed=rng,
        inversion=module.inversion,
    )
    truncation = module.enkf_truncation

    if module.localization:
        smoother_adaptive_es = AdaptiveESMDA(
            covariance=observation_errors**2,
            observations=observation_values,
            seed=rng,
        )

        # Pre-calculate cov_YY
        cov_YY = np.cov(S)

        D = smoother_adaptive_es.perturb_observations(
            ensemble_size=ensemble_size, alpha=1.0
        )

    else:
        # Compute transition matrix so that
        # X_posterior = X_prior @ T
        T = smoother_es.compute_transition_matrix(Y=S, alpha=1.0, truncation=truncation)
        # Add identity in place for fast computation
        np.fill_diagonal(T, T.diagonal() + 1)

    for param_group in parameters:
        source = source_ensemble
        temp_storage = _create_temporary_parameter_storage(
            source, iens_active_index, param_group
        )
        if module.localization:
            num_params = temp_storage[param_group].shape[0]
            batch_size = _calculate_adaptive_batch_size(num_params, num_obs)
            batches = _split_by_batchsize(np.arange(0, num_params), batch_size)

            log_msg = f"Running localization on {num_params} parameters, {num_obs} responses, {ensemble_size} realizations and {len(batches)} batches"
            _logger.info(log_msg)
            progress_callback(AnalysisStatusEvent(msg=log_msg))

            start = time.time()
            for param_batch_idx in batches:
                X_local = temp_storage[param_group][param_batch_idx, :]
                temp_storage[param_group][param_batch_idx, :] = (
                    smoother_adaptive_es.assimilate(
                        X=X_local,
                        Y=S,
                        D=D,
                        alpha=1.0,  # The user is responsible for scaling observation covariance (esmda usage)
                        correlation_threshold=module.correlation_threshold,
                        cov_YY=cov_YY,
                        progress_callback=adaptive_localization_progress_callback,
                    )
                )
            _logger.info(
                f"Adaptive Localization of {param_group} completed in {(time.time() - start) / 60} minutes"
            )

        else:
            temp_storage[param_group] = temp_storage[param_group] @ T.astype(
                temp_storage[param_group].dtype
            )

        log_msg = f"Storing data for {param_group}.."
        _logger.info(log_msg)
        progress_callback(AnalysisStatusEvent(msg=log_msg))
        start = time.time()
        _save_temp_storage_to_disk(target_ensemble, temp_storage, iens_active_index)
        _logger.info(
            f"Storing data for {param_group} completed in {(time.time() - start) / 60} minutes"
        )

        _copy_unupdated_parameters(
            list(source_ensemble.experiment.parameter_configuration.keys()),
            parameters,
            iens_active_index,
            source_ensemble,
            target_ensemble,
        )


def analysis_IES(
    parameters: Iterable[str],
    observations: Iterable[str],
    rng: np.random.Generator,
    analysis_config: IESSettings,
    alpha: float,
    std_cutoff: float,
    smoother_snapshot: SmootherSnapshot,
    ens_mask: npt.NDArray[np.bool_],
    source_ensemble: Ensemble,
    target_ensemble: Ensemble,
    sies_smoother: Optional[ies.SIES],
    progress_callback: Callable[[AnalysisEvent], None],
    auto_scale_observations: List[ObservationGroups],
    sies_step_length: Callable[[int], float],
    initial_mask: npt.NDArray[np.bool_],
) -> ies.SIES:
    iens_active_index = np.flatnonzero(ens_mask)
    # Pick out realizations that were among the initials that are still living
    # Example: initial_mask=[1,1,1,0,1], ens_mask=[0,1,1,0,1]
    # Then the result is [0,1,1,1]
    # This is needed for the SIES library
    masking_of_initial_parameters = ens_mask[initial_mask]

    progress_callback(AnalysisStatusEvent(msg="Loading observations and responses.."))

    (
        S,
        (
            observation_values,
            observation_errors,
            update_snapshot,
        ),
    ) = _load_observations_and_responses(
        source_ensemble,
        alpha,
        std_cutoff,
        1.0,
        iens_active_index,
        observations,
        auto_scale_observations,
    )

    smoother_snapshot.update_step_snapshots = update_snapshot
    if len(observation_values) == 0:
        msg = "No active observations for update step"
        progress_callback(
            AnalysisErrorEvent(error_msg=msg, smoother_snapshot=smoother_snapshot)
        )
        raise ErtAnalysisError(msg)

    # if the algorithm object is not passed, initialize it
    if sies_smoother is None:
        # The sies smoother must be initialized with the full parameter ensemble
        # Get relevant active realizations
        param_groups = list(source_ensemble.experiment.parameter_configuration.keys())
        parameter_ensemble_active = _all_parameters(
            source_ensemble, iens_active_index, param_groups
        )
        sies_smoother = ies.SIES(
            parameters=parameter_ensemble_active,
            covariance=observation_errors**2,
            observations=observation_values,
            seed=rng,
            inversion=analysis_config.inversion,
            truncation=analysis_config.enkf_truncation,
        )

        # Keep track of iterations to calculate step-lengths
        sies_smoother.iteration = 1

    # Calculate step-lengths to scale SIES iteration
    step_length = sies_step_length(sies_smoother.iteration)

    # Propose a transition matrix using only active realizations
    proposed_W = sies_smoother.propose_W_masked(
        S, ensemble_mask=masking_of_initial_parameters, step_length=step_length
    )

    # Store transition matrix for later use on sies object
    sies_smoother.W[:, masking_of_initial_parameters] = proposed_W

    for param_group in parameters:
        source = target_ensemble
        try:
            target_ensemble.load_parameters(group=param_group, realizations=0)["values"]
        except Exception:
            source = source_ensemble
        temp_storage = _create_temporary_parameter_storage(
            source, iens_active_index, param_group
        )
        X = temp_storage[param_group]
        temp_storage[param_group] = X + X @ sies_smoother.W / np.sqrt(
            len(iens_active_index) - 1
        )

        progress_callback(AnalysisStatusEvent(msg=f"Storing data for {param_group}.."))
        _save_temp_storage_to_disk(target_ensemble, temp_storage, iens_active_index)

    _copy_unupdated_parameters(
        list(source_ensemble.experiment.parameter_configuration.keys()),
        parameters,
        iens_active_index,
        source_ensemble,
        target_ensemble,
    )

    assert sies_smoother is not None, "sies_smoother should be initialized"

    # Increment the iteration number
    sies_smoother.iteration += 1

    # Return the sies smoother so it may be iterated over
    return sies_smoother


def _write_update_report(
    path: Path, snapshot: SmootherSnapshot, run_id: str, experiment: Experiment
) -> None:
    update_step = snapshot.update_step_snapshots

    (experiment._path / f"update_log_{run_id}.json").write_text(
        snapshot.model_dump_json()
    )

    fname = path / f"{run_id}.txt"
    fname.parent.mkdir(parents=True, exist_ok=True)
    update_step = snapshot.update_step_snapshots

    obs_info: DefaultDict[ObservationStatus, int] = defaultdict(lambda: 0)
    for update in update_step:
        obs_info[update.status] += 1

    with open(fname, "w", encoding="utf-8") as fout:
        fout.write("=" * 150 + "\n")
        timestamp = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
        fout.write(f"Time: {timestamp}\n")
        fout.write(f"Parent ensemble: {snapshot.source_ensemble_name}\n")
        fout.write(f"Target ensemble: {snapshot.target_ensemble_name}\n")
        fout.write(f"Alpha: {snapshot.alpha}\n")
        fout.write(f"Global scaling: {snapshot.global_scaling}\n")
        fout.write(f"Standard cutoff: {snapshot.std_cutoff}\n")
        fout.write(f"Run id: {run_id}\n")
        fout.write(f"Active observations: {obs_info[ObservationStatus.ACTIVE]}\n")
        fout.write(
            f"Deactivated observations - missing respons(es): {obs_info[ObservationStatus.MISSING_RESPONSE]}\n"
        )
        fout.write(
            f"Deactivated observations - ensemble_std > STD_CUTOFF: {obs_info[ObservationStatus.STD_CUTOFF]}\n"
        )
        fout.write(
            f"Deactivated observations - outlier: {obs_info[ObservationStatus.OUTLIER]}\n"
        )
        fout.write("-" * 150 + "\n")
        fout.write(
            "Observed history".rjust(56)
            + "|".rjust(17)
            + "Simulated data".rjust(32)
            + "|".rjust(13)
            + "Status".rjust(12)
            + "\n"
        )
        fout.write("-" * 150 + "\n")
        for nr, step in enumerate(update_step):
            obs_std = (
                f"{step.obs_std:.3f}"
                if step.obs_scaling == 1
                else f"{step.obs_std * step.obs_scaling:.3f} ({step.obs_std:<.3f} * {step.obs_scaling:.3f})"
            )
            fout.write(
                f"{nr+1:^6}: {step.obs_name:20} {step.obs_val:>16.3f} +/- "
                f"{obs_std:<21} | {step.response_mean:>21.3f} +/- "
                f"{step.response_std:<16.3f} {'|':<6} "
                f"{step.get_status().capitalize()}\n"
            )


def _assert_has_enough_realizations(
    ens_mask: npt.NDArray[np.bool_], min_required_realizations: int
) -> None:
    active_realizations = ens_mask.sum()
    if active_realizations < min_required_realizations:
        raise ErtAnalysisError(
            f"There are {active_realizations} active realisations left, which is "
            "less than the minimum specified - stopping assimilation.",
        )


def _create_smoother_snapshot(
    prior_name: str,
    posterior_name: str,
    analysis_config: UpdateSettings,
    global_scaling: float,
) -> SmootherSnapshot:
    return SmootherSnapshot(
        source_ensemble_name=prior_name,
        target_ensemble_name=posterior_name,
        alpha=analysis_config.alpha,
        std_cutoff=analysis_config.std_cutoff,
        global_scaling=global_scaling,
        update_step_snapshots=[],
    )


def smoother_update(
    prior_storage: Ensemble,
    posterior_storage: Ensemble,
    run_id: str,
    observations: Iterable[str],
    parameters: Iterable[str],
    analysis_config: Optional[UpdateSettings] = None,
    es_settings: Optional[ESSettings] = None,
    rng: Optional[np.random.Generator] = None,
    progress_callback: Optional[Callable[[AnalysisEvent], None]] = None,
    global_scaling: float = 1.0,
    log_path: Optional[Path] = None,
) -> SmootherSnapshot:
    if not progress_callback:
        progress_callback = noop_progress_callback
    if rng is None:
        rng = np.random.default_rng()
    analysis_config = UpdateSettings() if analysis_config is None else analysis_config
    es_settings = ESSettings() if es_settings is None else es_settings
    ens_mask = prior_storage.get_realization_mask_with_responses()
    _assert_has_enough_realizations(ens_mask, analysis_config.min_required_realizations)

    smoother_snapshot = _create_smoother_snapshot(
        prior_storage.name,
        posterior_storage.name,
        analysis_config,
        global_scaling,
    )

    try:
        analysis_ES(
            parameters,
            observations,
            rng,
            es_settings,
            analysis_config.alpha,
            analysis_config.std_cutoff,
            global_scaling,
            smoother_snapshot,
            ens_mask,
            prior_storage,
            posterior_storage,
            progress_callback,
            analysis_config.auto_scale_observations,
        )
    except Exception as e:
        raise e
    finally:
        if log_path is not None:
            _write_update_report(
                log_path,
                smoother_snapshot,
                run_id,
                prior_storage.experiment,
            )

    return smoother_snapshot


def iterative_smoother_update(
    prior_storage: Ensemble,
    posterior_storage: Ensemble,
    sies_smoother: Optional[ies.SIES],
    run_id: str,
    parameters: Iterable[str],
    observations: Iterable[str],
    update_settings: UpdateSettings,
    analysis_config: IESSettings,
    sies_step_length: Callable[[int], float],
    initial_mask: npt.NDArray[np.bool_],
    rng: Optional[np.random.Generator] = None,
    progress_callback: Optional[Callable[[AnalysisEvent], None]] = None,
    log_path: Optional[Path] = None,
    global_scaling: float = 1.0,
) -> Tuple[SmootherSnapshot, ies.SIES]:
    if not progress_callback:
        progress_callback = noop_progress_callback
    if rng is None:
        rng = np.random.default_rng()

    ens_mask = prior_storage.get_realization_mask_with_responses()
    _assert_has_enough_realizations(ens_mask, update_settings.min_required_realizations)

    smoother_snapshot = _create_smoother_snapshot(
        prior_storage.name,
        posterior_storage.name,
        update_settings,
        global_scaling,
    )

    try:
        sies_smoother = analysis_IES(
            parameters=parameters,
            observations=observations,
            rng=rng,
            analysis_config=analysis_config,
            alpha=update_settings.alpha,
            std_cutoff=update_settings.std_cutoff,
            smoother_snapshot=smoother_snapshot,
            ens_mask=ens_mask,
            source_ensemble=prior_storage,
            target_ensemble=posterior_storage,
            sies_smoother=sies_smoother,
            progress_callback=progress_callback,
            auto_scale_observations=update_settings.auto_scale_observations,
            sies_step_length=sies_step_length,
            initial_mask=initial_mask,
        )
    except Exception as e:
        raise e
    finally:
        if log_path is not None:
            _write_update_report(
                log_path,
                smoother_snapshot,
                run_id,
                prior_storage.experiment,
            )

    return smoother_snapshot, sies_smoother
