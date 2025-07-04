from __future__ import annotations

import functools
import logging
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import (
    TYPE_CHECKING,
    Generic,
    Self,
    TextIO,
    TypeVar,
)

import iterative_ensemble_smoother as ies
import numpy as np
import polars as pl
import psutil
import scipy
from iterative_ensemble_smoother.experimental import AdaptiveESMDA
from threadpoolctl import threadpool_limits

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

OPTIMAL_NUM_THREADS = max(1, (psutil.cpu_count(logical=False) or 1) // 2)


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
        logger.info(
            f"Will run Adaptive Localization using {OPTIMAL_NUM_THREADS} threads."
        )
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

        log_msg = f"There are {num_obs} responses and {ensemble_size} realizations."
        logger.info(log_msg)
        progress_callback(AnalysisStatusEvent(msg=log_msg))

        log_msg = (
            f"There are {(~non_zero_variance_mask).sum()} parameters with 0 variance "
            f"that will not be updated."
        )
        logger.info(log_msg)
        progress_callback(AnalysisStatusEvent(msg=log_msg))

        if module.localization:
            with threadpool_limits(limits=OPTIMAL_NUM_THREADS):
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
                    update_idx = param_batch_idx[
                        non_zero_variance_mask[param_batch_idx]
                    ]
                    X_local = param_ensemble_array[update_idx, :]
                    if isinstance(config_node, GenKwConfig):
                        correlation_batch_callback = functools.partial(
                            correlation_callback,
                            cross_correlations_accumulator=cross_correlations,
                        )
                    else:
                        correlation_batch_callback = None
                    param_ensemble_array[update_idx, :] = (
                        smoother_adaptive_es.assimilate(
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
                        )
                    )

                if cross_correlations:
                    assert isinstance(config_node, GenKwConfig)
                    parameter_names = config_node.parameter_keys
                    cross_correlations_ = np.vstack(cross_correlations)
                    if cross_correlations_.size != 0:
                        source_ensemble.save_cross_correlations(
                            cross_correlations_,
                            param_group,
                            parameter_names[: cross_correlations_.shape[0]],
                        )
                logger.info(
                    f"Adaptive Localization of {param_group} completed "
                    f"in {(time.time() - start) / 60} minutes"
                )

        else:
            # In-place multiplication is not yet supported, therefore avoiding @=
            param_ensemble_array[non_zero_variance_mask] = param_ensemble_array[  # noqa: PLR6104
                non_zero_variance_mask
            ] @ T.astype(param_ensemble_array.dtype)

        log_msg = f"Storing data for {param_group}.."
        logger.info(log_msg)
        progress_callback(AnalysisStatusEvent(msg=log_msg))
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
) -> SmootherSnapshot:
    if not progress_callback:
        progress_callback = noop_progress_callback
    if rng is None:
        rng = np.random.default_rng()

    ens_mask = prior_storage.get_realization_mask_with_responses()

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

            def log_warning(
                message: Warning | str,
                category: type[Warning],
                filename: str,
                lineno: int,
                file: TextIO | None = None,
                line: str | None = None,
            ) -> None:
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
