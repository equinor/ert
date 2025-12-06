from __future__ import annotations

import functools
import logging
import os
import re
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, Generic, Self, TextIO, TypeVar

import iterative_ensemble_smoother as ies
import numpy as np
import polars as pl
import psutil
import scipy
from iterative_ensemble_smoother.experimental import AdaptiveESMDA, DistanceESMDA

from ert.config import (
    ESSettings,
    GenKwConfig,
    ObservationSettings,
)

from ._update_commons import (
    ErtAnalysisError,
    _copy_unupdated_parameters,
    _OutlierColumns,
    _preprocess_observations_and_responses,
    noop_progress_callback,
)
from .event import (
    AnalysisCompleteEvent,
    AnalysisErrorEvent,
    AnalysisEvent,
    AnalysisStatusEvent,
    AnalysisTimeEvent,
    DataSection,
)
from .snapshots import (
    ObservationStatus,
    SmootherSnapshot,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble

logger = logging.getLogger(__name__)

T = TypeVar("T")

# When running adaptive localization we reserve some resources for
# the GUI and other applications that might be running
RESERVED_CPU_CORES = 2
# Testing on drogon seems to indicate that this is a
# reasonable 'default' value for the parallel config of joblib
NUM_JOBS_ADAPTIVE_LOC = max(
    1, ((psutil.cpu_count(logical=False) or 1) - RESERVED_CPU_CORES)
)


class TimedIterator(Generic[T]):
    SEND_FREQUENCY = 1.0  # seconds

    def __init__(
        self, iterable: Sequence[T], callback: Callable[[AnalysisEvent], None]
    ) -> None:
        self._start_time = time.perf_counter()
        self._iterable = iterable
        self._callback = callback
        self._index = 0
        self._last_send_time = 0.0

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
            if elapsed_time - self._last_send_time > self.SEND_FREQUENCY:
                self._callback(
                    AnalysisTimeEvent(
                        remaining_time=estimated_remaining_time,
                        elapsed_time=elapsed_time,
                    )
                )
                self._last_send_time = elapsed_time

        self._index += 1
        return result


def _split_by_batchsize(
    arr: npt.NDArray[np.int_], batch_size: int
) -> list[npt.NDArray[np.int_]]:
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
    """Calculate adaptive batch size to optimize memory usage during Adaptive
    Localization. Adaptive Localization calculates the cross-covariance between
    parameters and responses. Cross-covariance is a matrix with shape num_params
    x num_obs which may be larger than memory. Therefore, a batching algorithm is
    used where only a subset of parameters is used when calculating cross-covariance.
    This function calculates a batch size that can fit into the available memory,
    accounting for a safety margin.

    Derivation of formula:
    ---------------------
    available_memory = (amount of available memory on system) * memory_safety_factor
    required_memory = num_params * num_obs * bytes_in_float32
    num_params = required_memory / (num_obs * bytes_in_float32)
    We want (required_memory < available_memory) so:
    num_params < available_memory / (num_obs * bytes_in_float32)

    The available memory is checked using the `psutil` library, which provides
    information about system memory usage.
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


def calc_max_number_of_layers_per_batch_for_distance_localization(
    nx: int,
    ny: int,
    nz: int,
    num_obs: int,
    nreal: int,
    bytes_per_float: int = 8,  # float64 as default here
) -> int:
    """Calculate number of layers from a 3D field parameter that can be updated
    within available memory. Distance-based localization requires two large matrices
    the Kalman gain matrix K and the localization scaling matrix RHO, both have size
    equal to number of field parameter values times number of observations.
    Therefore, a batching algorithm is used where only a subset of parameters
    is used when calculating the Schur product of RHO and K matrix in the update
    algorithm. This function calculates number of batches and
    number of grid layers of field parameter values that can fit
    into the available memory for one batch accounting for a safety margin.

    The available memory is checked using the `psutil` library, which provides
    information about system memory usage.
    From `psutil` documentation:
    - available:
        the memory that can be given instantly to processes without the
        system going into swap.
        This is calculated by summing different memory values depending
        on the platform and it is supposed to be used to monitor actual
        memory usage in a cross platform fashion.

    Args:
        nx: grid size in I-direction (local x-axis direction)
        ny: grid size in J-direction (local y-axis direction)
        nz: grid size in K-direction (number of layers)
        num_obs: Number of observations
        nreal: Number of realizations
        bytes_per_float: Is 4 or 8

    Returns:
        Max number of layers that can be updated in one batch to
        avoid memory problems.

    """

    memory_safety_factor = 0.8
    num_params = nx * ny * nz
    num_param_per_layer = nx * ny

    # Rough estimate of necessary number of float variables
    sum_floats = 0
    sum_floats += num_params * num_obs  # K matrix before Schur product
    sum_floats += num_params * num_obs  # RHO matrix
    sum_floats += num_params * num_obs  # K matrix after Schur product
    sum_floats += int(num_params * nreal * 2.5)  # X_prior, X_prior_batch, M_delta
    sum_floats += int(num_params * nreal * 1.5)  # X_post and X_post_batch
    sum_floats += num_obs * nreal * 2  # D matrix and internal matrices
    sum_floats += num_obs * nreal * 2  # Y matrix and internal matrices

    # Check available memory
    available_memory_in_bytes = psutil.virtual_memory().available * memory_safety_factor

    # Required memory
    total_required_memory_per_field_param = sum_floats * bytes_per_float

    # Minimum number of batches
    min_number_of_batches = int(
        np.ceil(total_required_memory_per_field_param / available_memory_in_bytes)
    )

    max_nlayer_per_batch = int(nz / min_number_of_batches)

    if max_nlayer_per_batch == 0:
        # Batch size cannot be less than 1 layer
        memory_one_batch = num_param_per_layer * bytes_per_float
        raise MemoryError(
            "The required memory to update one grid layer or one 2D surface is "
            "larger than available memory.\n"
            "Cannot split the update into batch size less than one complete "
            "grid layer for 3D field or one surface for 2D fields."
            f"Required memory for one batch is about: {memory_one_batch / 10**9} GB\n"
            f"Available memory is about: {available_memory_in_bytes / 10**9} GB"
        )

    log_msg = (
        "Calculate batch size for updating of field parameter:\n"
        f" Number of parameters in field param: {num_params}\n"
        f" Required number of floats to update one field parameter: {sum_floats}\n"
        " Available memory per field param update: "
        f"{available_memory_in_bytes / 10**9} GB\n"
        " Required memory total to update a field parameter: "
        f"{total_required_memory_per_field_param / 10**9} GB\n"
        f" Number of layers in one batch: {max_nlayer_per_batch}"
    )
    logger.info(log_msg)
    return max_nlayer_per_batch


def analysis_ES(
    parameters: Iterable[str],
    observations: Iterable[str],
    rng: np.random.Generator,
    module: ESSettings,
    observation_settings: ObservationSettings,
    global_scaling: float,
    smoother_snapshot: SmootherSnapshot,
    ens_mask: npt.NDArray[np.bool_],
    source_ensemble: Ensemble,
    target_ensemble: Ensemble,
    progress_callback: Callable[[AnalysisEvent], None],
) -> None:
    iens_active_index = np.flatnonzero(ens_mask)

    ensemble_size = ens_mask.sum()

    def adaptive_localization_progress_callback(
        iterable: Sequence[T],
    ) -> TimedIterator[T]:
        return TimedIterator(iterable, progress_callback)

    preprocessed_data = _preprocess_observations_and_responses(
        ensemble=source_ensemble,
        outlier_settings=observation_settings.outlier_settings,
        auto_scale_observations=observation_settings.auto_scale_observations,
        iens_active_index=iens_active_index,
        global_std_scaling=global_scaling,
        selected_observations=observations,
        progress_callback=progress_callback,
    )

    filtered_data = preprocessed_data.filter(
        pl.col("status") == ObservationStatus.ACTIVE
    )

    S = filtered_data.select([*map(str, iens_active_index)]).to_numpy(order="c")
    observation_values = filtered_data["observations"].to_numpy()
    observation_errors = filtered_data[_OutlierColumns.scaled_std].to_numpy()

    progress_callback(AnalysisStatusEvent(msg="Loading observations and responses.."))
    num_obs = len(observation_values)

    smoother_snapshot.observations_and_responses = preprocessed_data.drop(
        [*map(str, iens_active_index), "response_key"]
    ).select(
        "observation_key",
        "index",
        "observations",
        "std",
        "obs_error_scaling",
        "scaled_obs_error",
        "response_mean",
        "response_std",
        "status",
    )

    if num_obs == 0:
        msg = "No active observations for update step"
        progress_callback(
            AnalysisErrorEvent(
                error_msg=msg,
                data=DataSection(
                    header=smoother_snapshot.header,
                    data=smoother_snapshot.csv,
                    extra=smoother_snapshot.extra,
                ),
            )
        )
        raise ErtAnalysisError(msg)

    smoother_es = ies.ESMDA(
        covariance=observation_errors**2,
        observations=observation_values,
        # The user is responsible for scaling observation covariance (esmda usage):
        alpha=1,
        seed=rng,
        inversion=module.inversion.lower(),
    )
    truncation = module.enkf_truncation

    if module.localization:
        smoother_adaptive_es = AdaptiveESMDA(
            covariance=observation_errors**2,
            observations=observation_values,
            seed=rng,
        )

        # Pre-calculate cov_YY
        cov_YY = np.atleast_2d(np.cov(S))

        D = smoother_adaptive_es.perturb_observations(
            ensemble_size=ensemble_size, alpha=1.0
        )

    else:
        # Compute transition matrix so that
        # X_posterior = X_prior @ T
        try:
            T = smoother_es.compute_transition_matrix(
                Y=S, alpha=1.0, truncation=truncation
            )
        except scipy.linalg.LinAlgError as err:
            msg = (
                "Failed while computing transition matrix, "
                "this might be due to outlier values in one "
                f"or more realizations: {err}"
            )
            progress_callback(
                AnalysisErrorEvent(
                    error_msg=msg,
                    data=DataSection(
                        header=smoother_snapshot.header,
                        data=smoother_snapshot.csv,
                        extra=smoother_snapshot.extra,
                    ),
                )
            )
            raise ErtAnalysisError(msg) from err
        # Add identity in place for fast computation
        np.fill_diagonal(T, T.diagonal() + 1)

    def correlation_callback(
        cross_correlations_of_batch: npt.NDArray[np.float64],
        cross_correlations_accumulator: list[npt.NDArray[np.float64]],
    ) -> None:
        cross_correlations_accumulator.append(cross_correlations_of_batch)

    for param_group in parameters:
        param_ensemble_array = source_ensemble.load_parameters_numpy(
            param_group, iens_active_index
        )

        # Calculate variance for each parameter
        param_variance = np.var(param_ensemble_array, axis=1)
        # Create mask for non-zero variance parameters
        non_zero_variance_mask = ~np.isclose(param_variance, 0.0)

        log_msg = (
            f"Updating {np.sum(non_zero_variance_mask)} parameters "
            f"{'with' if module.localization else 'without'} "
            f"adaptive localization."
        )
        logger.info(log_msg)
        progress_callback(AnalysisStatusEvent(msg=log_msg))

        if (param_count := (~non_zero_variance_mask).sum()) > 0:
            log_msg = (
                f"There are {param_count} parameters with 0 variance "
                f"that will not be updated."
            )
            logger.info(log_msg)
            progress_callback(AnalysisStatusEvent(msg=log_msg))

        if module.localization:
            config_node = source_ensemble.experiment.parameter_configuration[
                param_group
            ]
            num_params = param_ensemble_array.shape[0]
            batch_size = _calculate_adaptive_batch_size(num_params, num_obs)
            batches = _split_by_batchsize(np.arange(0, num_params), batch_size)

            log_msg = (
                f"Running localization on {num_params} parameters, "
                f"{num_obs} responses, {ensemble_size} realizations "
                f"and {len(batches)} batches"
            )
            logger.info(log_msg)
            progress_callback(AnalysisStatusEvent(msg=log_msg))

            start = time.time()
            cross_correlations: list[npt.NDArray[np.float64]] = []
            for param_batch_idx in batches:
                update_idx = param_batch_idx[non_zero_variance_mask[param_batch_idx]]
                X_local = param_ensemble_array[update_idx, :]
                if isinstance(config_node, GenKwConfig):
                    correlation_batch_callback = functools.partial(
                        correlation_callback,
                        cross_correlations_accumulator=cross_correlations,
                    )
                else:
                    correlation_batch_callback = None
                param_ensemble_array[update_idx, :] = smoother_adaptive_es.assimilate(
                    X=X_local,
                    Y=S,
                    D=D,
                    # The user is responsible for scaling observation covariance
                    # (ESMDA usage)
                    alpha=1.0,
                    correlation_threshold=module.correlation_threshold,
                    cov_YY=cov_YY,
                    progress_callback=adaptive_localization_progress_callback,
                    correlation_callback=correlation_batch_callback,
                    # number of parallel jobs for joblib
                    n_jobs=NUM_JOBS_ADAPTIVE_LOC,
                )
            logger.info(
                f"Adaptive Localization of {param_group} completed "
                f"in {(time.time() - start) / 60} minutes"
            )

        else:
            log_msg = f"There are {num_obs} responses and {ensemble_size} realizations."
            logger.info(log_msg)
            progress_callback(AnalysisStatusEvent(msg=log_msg))

            # In-place multiplication is not yet supported, therefore avoiding @=
            param_ensemble_array[non_zero_variance_mask] = param_ensemble_array[  # noqa: PLR6104
                non_zero_variance_mask
            ] @ T.astype(param_ensemble_array.dtype)

        start = time.time()
        target_ensemble.save_parameters_numpy(
            param_ensemble_array, param_group, iens_active_index
        )
        logger.info(
            f"Storing data for {param_group} completed in "
            f"{(time.time() - start) / 60} minutes"
        )

    _copy_unupdated_parameters(
        list(source_ensemble.experiment.parameter_configuration.keys()),
        parameters,
        iens_active_index,
        source_ensemble,
        target_ensemble,
    )


def smoother_update(
    prior_storage: Ensemble,
    posterior_storage: Ensemble,
    observations: Iterable[str],
    parameters: Iterable[str],
    update_settings: ObservationSettings,
    es_settings: ESSettings,
    rng: np.random.Generator | None = None,
    progress_callback: Callable[[AnalysisEvent], None] | None = None,
    global_scaling: float = 1.0,
    active_realizations: list[bool] | None = None,
) -> SmootherSnapshot:
    if not progress_callback:
        progress_callback = noop_progress_callback
    if rng is None:
        rng = np.random.default_rng()

    ens_mask = prior_storage.get_realization_mask_with_responses()
    if active_realizations:
        ens_mask &= active_realizations

    smoother_snapshot = SmootherSnapshot(
        source_ensemble_name=prior_storage.name,
        target_ensemble_name=posterior_storage.name,
        alpha=update_settings.outlier_settings.alpha,
        std_cutoff=update_settings.outlier_settings.std_cutoff,
        global_scaling=global_scaling,
    )

    try:
        with warnings.catch_warnings():
            original_showwarning = warnings.showwarning

            ILL_CONDITIONED_RE = re.compile(
                r"^LinAlgWarning:.*ill[- ]?conditioned\s+matrix", re.IGNORECASE
            )
            LIMIT_ILL_CONDITIONED_WARNING = 1000
            illconditioned_warn_counter = 0

            def log_warning(
                message: Warning | str,
                category: type[Warning],
                filename: str,
                lineno: int,
                file: TextIO | None = None,
                line: str | None = None,
            ) -> None:
                nonlocal illconditioned_warn_counter

                if ILL_CONDITIONED_RE.search(str(message)):
                    illconditioned_warn_counter += 1

                if illconditioned_warn_counter < LIMIT_ILL_CONDITIONED_WARNING:
                    logger.warning(
                        f"{category.__name__}: {message} (from {filename}:{lineno})"
                    )
                    original_showwarning(
                        message, category, filename, lineno, file=file, line=line
                    )

            warnings.showwarning = log_warning
            analysis_ES(
                parameters,
                observations,
                rng,
                es_settings,
                update_settings,
                global_scaling,
                smoother_snapshot,
                ens_mask,
                prior_storage,
                posterior_storage,
                progress_callback,
            )
    except Exception as e:
        progress_callback(
            AnalysisErrorEvent(
                error_msg=str(e),
                data=DataSection(
                    header=smoother_snapshot.header,
                    data=smoother_snapshot.csv,
                    extra=smoother_snapshot.extra,
                ),
            )
        )
        raise e
    progress_callback(
        AnalysisCompleteEvent(
            data=DataSection(
                header=smoother_snapshot.header,
                data=smoother_snapshot.csv,
                extra=smoother_snapshot.extra,
            )
        )
    )
    return smoother_snapshot


def memory_usage_decorator(
    enabled: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:  # type: ignore
            if enabled:
                # Get process ID of the current Python process
                process = psutil.Process(os.getpid())

                # Memory usage before the function call
                mem_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
                print(f"Memory before calling '{func.__name__}': {mem_before:.2f} MB")
                start_time = time.perf_counter()
            # Call the target function
            result = func(*args, **kwargs)

            if enabled:
                end_time = time.perf_counter()
                # Memory usage after the function call
                mem_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
                print(f"Memory after calling '{func.__name__}': {mem_after:.2f} MB")
                print(f"Run time used: {(end_time - start_time):.4f} seconds")
            return result

        return wrapper

    return decorator


@memory_usage_decorator(enabled=False)
def update_3D_field_with_distance_esmda(
    distance_based_esmda_smoother: DistanceESMDA,
    field_param_name: str,
    X_prior: npt.NDArray[np.float64],
    Y: npt.NDArray[np.float64],
    rho_2D: npt.NDArray[np.float64],
    nx: int,
    ny: int,
    nz: int,
    reshape_to_3D_per_realization: bool = False,
    min_nbatch: int = 1,
) -> npt.NDArray[np.float64]:
    """
    Calculate posterior update with distance-based ESMDA for one 3D parameter
    The RHO for one layer of the 3D field parameter is input.
    This is copied to all other layers of RHO in each batch of grid parameter
    layers since only lateral distance is used when calculating distances.
    Result is posterior parameter matrices of field parameters for one field.

    Args:
        distance_based_esmda_smooter: Object of DistanceESMDA class initialized for use
        field_param_name: Name of 3D parameter
        X_prior: Matrix with prior realizations of all field parameters,
                 shape=(nparameters, nrealizations)
        Y: Matrix with response values for each observations for each realization,
                 shape=(nobservations, nrealizations)
        rho_2D: RHO matrix elements for one 3D grid layer with size (nx, ny),
                 shape=(nx,ny,nobservations)
        nx, ny, nz: Dimensions of the 3D grid filled with a 3D field parameter.
        reshape_to_3D_per_realization: Is set to True if output field is reshaped to 3D
                  per realization.
        min_nbatch: Minimum number of batches the field parameter is split into.
                  Default is 1. Usually number of batches will be calculated based
                  on available memory and the size of the field parameters,
                  number of observations and realizations. The actual number of
                  batches will be max(min_nbatch, min_number_of_batches_required).

    Results:
        X_post: Posterior ensemble of field parameters,
          shape=(nx*ny*nz, nrealizations) if not reshaped and
          shape=(nx,ny,nz,nrealizaions) if reshaped to 3D per realization
    """
    nparam_per_layer = nx * ny
    nparam = nparam_per_layer * nz
    nreal = X_prior.shape[1]
    assert X_prior.shape[0] == nparam, (
        f"Mismatch between X_prior dimension {X_prior.shape[0]} and nparam {nparam}"
    )

    X_prior_3D = X_prior.reshape(nx, ny, nz, nreal)
    # No update if no observations or responses
    if Y is None or Y.shape[0] == 0:
        # No update of the field parameters
        # Check if it necessary to make a copy or can we only return X_prior?
        if reshape_to_3D_per_realization:
            return X_prior_3D.copy()
        return X_prior.copy()

    nobs = Y.shape[0]
    assert Y.shape[1] == nreal, (
        f"Mismatch between X_prior dimension {Y.shape[1]} and nreal {nreal}"
    )

    log_msg = f"Calculate Distance-based ESMDA update for {field_param_name} "
    log_msg += f"with {nparam} parameters"
    logger.info(log_msg)

    # Check memory constraints and calculate how many grid layers of
    # field parameters is possible to update on one batch
    max_nlayers_per_batch = (
        calc_max_number_of_layers_per_batch_for_distance_localization(
            nx, ny, nz, nobs, nreal, bytes_per_float=8
        )
    )  # Use float64
    nlayer_per_batch = min(max_nlayers_per_batch, nz)
    nbatch = int(nz / nlayer_per_batch)

    log_msg = "Minimum number of batches required due to "
    log_msg += f"limited memory constraints: {nbatch}"
    logger.info(log_msg)

    # Number of batches is defined by available memory and wanted number of batches
    # Usually one should have as few batches as possible but sufficient to
    # be able to update a batch with the memory available.
    # It is possible to explicit require more batches than the minimum
    # number required by the memory constraint. Main use case for this
    # is probably only for unit testing to avoid having unit tests running
    # slow due to very big size of the field, the number of observations and
    # and number of realizations.
    nbatch = max(min_nbatch, nbatch)
    nlayer_per_batch = int(nz / nbatch)
    nparam_in_batch = (
        nparam_per_layer * nlayer_per_batch
    )  # For full sized batch of layers

    nlayer_last_batch = nz - nbatch * nlayer_per_batch
    if nlayer_last_batch > 0:
        log_msg = f"Number of batches to update {field_param_name} is {nbatch + 1}"
        logger.info(log_msg)
    else:
        log_msg = f"Number of batches to update {field_param_name} is {nbatch}"
        logger.info(log_msg)

    X_prior_3D = X_prior.reshape(nx, ny, nz, nreal)

    # Initialize the X_post_3D
    X_post_3D = X_prior_3D.copy()
    for batch_number in range(nbatch):
        start_layer_number = batch_number * nlayer_per_batch
        end_layer_number = start_layer_number + nlayer_per_batch
        log_msg = (
            f"Batch number: {batch_number}\n"
            f"start layer : {start_layer_number}\n"
            f"end layer   : {end_layer_number - 1}"
        )
        logger.info(log_msg)

        X_batch = X_prior_3D[:, :, start_layer_number:end_layer_number, :].reshape(
            (nparam_in_batch, nreal)
        )

        # Copy rho calculated from one layer of 3D parameter into all layers for
        # current batch of layers.
        # Size of rho batch: (nx,ny,nlayer_per_batch,nobs)
        rho_3D_batch = np.zeros((nx, ny, nlayer_per_batch, nobs), dtype=np.float64)
        rho_3D_batch[:, :, :, :] = rho_2D[:, :, np.newaxis, :]
        rho_batch = rho_3D_batch.reshape((nparam_in_batch, nobs))

        log_msg = f"Assimilate batch number {batch_number}"
        logger.info(log_msg)
        X_post_batch = distance_based_esmda_smoother.assimilate_batch(
            X_batch=X_batch, Y=Y, rho_batch=rho_batch
        )
        X_post_3D[:, :, start_layer_number:end_layer_number, :] = X_post_batch.reshape(
            nx, ny, nlayer_per_batch, nreal
        )

    if nlayer_last_batch > 0:
        batch_number = nbatch
        start_layer_number = batch_number * nlayer_per_batch
        end_layer_number = start_layer_number + nlayer_last_batch
        nparam_in_last_batch = nparam_per_layer * nlayer_last_batch
        log_msg = f"Batch number: {batch_number}\n"
        log_msg += f"start layer : {start_layer_number}\n"
        log_msg += f"end layer   : {end_layer_number - 1}"
        logger.info(log_msg)

        X_batch = X_prior_3D[:, :, start_layer_number:end_layer_number, :].reshape(
            (nparam_in_last_batch, nreal)
        )

        rho_3D_batch = np.zeros((nx, ny, nlayer_last_batch, nobs), dtype=np.float64)
        # Copy rho calculated from one layer of 3D parameter into all layers for
        # current batch of layers
        rho_3D_batch[:, :, :, :] = rho_2D[:, :, np.newaxis, :]
        rho_batch = rho_3D_batch.reshape((nparam_in_last_batch, nobs))

        log_msg = f"Assimilate batch number {batch_number}"
        logger.info(log_msg)
        X_post_batch = distance_based_esmda_smoother.assimilate_batch(
            X_batch=X_batch, Y=Y, rho_batch=rho_batch
        )
        X_post_3D[:, :, start_layer_number:end_layer_number, :] = X_post_batch.reshape(
            nx, ny, nlayer_last_batch, nreal
        )
    if reshape_to_3D_per_realization:
        return X_post_3D
    else:
        return X_post_3D.reshape(nparam, nreal)
