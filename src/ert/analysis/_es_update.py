from __future__ import annotations

import logging
import re
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Generic, Self, TextIO, TypeVar

import iterative_ensemble_smoother as ies
import numpy as np
import polars as pl
import psutil
import scipy
from iterative_ensemble_smoother.experimental import AdaptiveESMDA, DistanceESMDA
from iterative_ensemble_smoother.utils import calc_rho_for_2d_grid_layer

from ert.config import (
    ESSettings,
    Field,
    GenKwConfig,
    ObservationSettings,
    SurfaceConfig,
)
from ert.field_utils import (
    AxisOrientation,
    transform_local_ellipse_angle_to_local_coords,
    transform_positions_to_local_field_coordinates,
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
from .snapshots import ObservationStatus, SmootherSnapshot

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

    obs_xpos: npt.NDArray[np.float64] | None = None
    obs_ypos: npt.NDArray[np.float64] | None = None
    obs_main_range: npt.NDArray[np.float64] | None = None
    S_with_loc: npt.NDArray[np.float64] | None = None
    smoother_distance_es: DistanceESMDA | None = None
    if module.distance_localization:
        # get observations with location
        has_location = (
            filtered_data["east"].is_not_null() & filtered_data["north"].is_not_null()
        ).to_numpy()

        obs_values_with_loc = filtered_data["observations"].to_numpy()[has_location]
        obs_errors_with_loc = filtered_data[_OutlierColumns.scaled_std].to_numpy()[
            has_location
        ]
        obs_xpos = filtered_data["east"].to_numpy()[has_location]
        obs_ypos = filtered_data["north"].to_numpy()[has_location]
        obs_main_range = filtered_data["radius"].to_numpy()[has_location]
        S_with_loc = S[has_location, :]

        smoother_distance_es = DistanceESMDA(
            covariance=obs_errors_with_loc**2,
            observations=obs_values_with_loc,
            alpha=1,
            seed=rng,
        )
    elif module.localization:
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
    if module.localization is False:
        # in case of distance localization we still
        # need to update the scalars
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

    for param_group in parameters:
        param_cfg = source_ensemble.experiment.parameter_configuration[param_group]
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

        if module.distance_localization:
            assert obs_xpos is not None
            assert obs_ypos is not None
            assert obs_main_range is not None
            assert smoother_distance_es is not None
            assert S_with_loc is not None

            if (
                isinstance(param_cfg, Field)
                and param_cfg.ertbox_params.origin is not None
            ):
                start = time.time()
                log_msg = (
                    f"Running distance localization on Field "
                    f" with {param_ensemble_array.shape[0]} parameters, "
                    f"{obs_xpos.shape[0]} observations, {ensemble_size} realizations "
                )
                logger.info(log_msg)

                if param_cfg.ertbox_params.axis_orientation is None:
                    logger.warning("Axis orientation is not defined, do not update")
                    continue

                assert param_cfg.ertbox_params.xinc is not None, (
                    "Parameter for grid resolution must be defined"
                )
                assert param_cfg.ertbox_params.yinc is not None, (
                    "Parameter for grid resolution must be defined"
                )
                assert param_cfg.ertbox_params.origin is not None, (
                    "Parameter for grid origin must be defined"
                )
                assert param_cfg.ertbox_params.rotation_angle is not None, (
                    "Parameter for grid rotation must be defined"
                )

                xpos, ypos = transform_positions_to_local_field_coordinates(
                    param_cfg.ertbox_params.origin,
                    param_cfg.ertbox_params.rotation_angle,
                    obs_xpos,
                    obs_ypos,
                )
                ellipse_rotation = transform_local_ellipse_angle_to_local_coords(
                    param_cfg.ertbox_params.rotation_angle,
                    np.zeros_like(obs_main_range, dtype=np.float64),
                )

                rho_matrix = calc_rho_for_2d_grid_layer(
                    param_cfg.ertbox_params.nx,
                    param_cfg.ertbox_params.ny,
                    param_cfg.ertbox_params.xinc,
                    param_cfg.ertbox_params.yinc,
                    xpos,
                    ypos,
                    obs_main_range,
                    obs_main_range,
                    ellipse_rotation,
                    param_cfg.ertbox_params.axis_orientation
                    == AxisOrientation.RIGHT_HANDED,
                )
                # right_handed - this needs to be retrieved from the grid
                param_ensemble_array = smoother_distance_es.update_params(
                    X=param_ensemble_array,
                    Y=S_with_loc,
                    rho_input=rho_matrix,
                    nz=param_cfg.ertbox_params.nz,
                )
                logger.info(
                    f"Distance Localization of Field {param_group} completed "
                    f"in {(time.time() - start) / 60} minutes"
                )
            elif isinstance(param_cfg, SurfaceConfig):
                start = time.time()
                log_msg = (
                    f"Running distance localization on Surface "
                    f" with {param_ensemble_array.shape[0]} parameters, "
                    f"{obs_xpos.shape[0]} observations, {ensemble_size} realizations "
                )
                xpos, ypos = transform_positions_to_local_field_coordinates(
                    (param_cfg.xori, param_cfg.yori),
                    param_cfg.rotation,
                    obs_xpos,
                    obs_ypos,
                )
                # Transform ellipse orientation to local surface coordinates
                rotation_angle_of_localization_ellipse = (
                    transform_local_ellipse_angle_to_local_coords(
                        param_cfg.rotation,
                        np.zeros_like(obs_main_range, dtype=np.float64),
                    )
                )
                assert param_cfg.yflip == 1
                rho_matrix = calc_rho_for_2d_grid_layer(
                    param_cfg.ncol,
                    param_cfg.nrow,
                    param_cfg.xinc,
                    param_cfg.yinc,
                    xpos,
                    ypos,
                    obs_main_range,
                    obs_main_range,
                    rotation_angle_of_localization_ellipse,
                    right_handed_grid_indexing=False,
                )
                param_ensemble_array = smoother_distance_es.update_params(
                    X=param_ensemble_array,
                    Y=S_with_loc,
                    rho_input=rho_matrix,
                )
                logger.info(
                    f"Distance Localization of Surface {param_group} completed "
                    f"in {(time.time() - start) / 60} minutes"
                )
        elif module.localization:
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
            for param_batch_idx in batches:
                update_idx = param_batch_idx[non_zero_variance_mask[param_batch_idx]]
                X_local = param_ensemble_array[update_idx, :]
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
                    # number of parallel jobs for joblib
                    n_jobs=NUM_JOBS_ADAPTIVE_LOC,
                )
            logger.info(
                f"Adaptive Localization of {param_group} completed "
                f"in {(time.time() - start) / 60} minutes"
            )

        if (
            module.distance_localization is False
            or (module.distance_localization and isinstance(param_cfg, GenKwConfig))
        ) and module.localization is False:
            log_msg = f"There are {num_obs} responses and {ensemble_size} realizations."
            logger.info(log_msg)
            progress_callback(AnalysisStatusEvent(msg=log_msg))

            # In-place multiplication is not yet supported, therefore avoiding @=
            param_ensemble_array[non_zero_variance_mask] @= T.astype(
                param_ensemble_array.dtype
            )

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
