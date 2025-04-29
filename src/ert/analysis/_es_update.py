from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from fnmatch import fnmatch
from typing import (
    TYPE_CHECKING,
    Generic,
    Self,
    TypeVar,
)

import iterative_ensemble_smoother as ies
import numpy as np
import polars as pl
import psutil
import scipy
from iterative_ensemble_smoother.experimental import AdaptiveESMDA

from ert.config import (
    ESSettings,
    GenKwConfig,
    ObservationGroups,
    ObservationSettings,
    OutlierSettings,
)

from . import misfit_preprocessor
from .event import (
    AnalysisCompleteEvent,
    AnalysisDataEvent,
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


class ErtAnalysisError(Exception):
    pass


def noop_progress_callback(_: AnalysisEvent) -> None:
    pass


T = TypeVar("T")


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


def _all_parameters(
    ensemble: Ensemble,
    iens_active_index: npt.NDArray[np.int_],
) -> npt.NDArray[np.float64]:
    """Return all parameters in assimilation problem"""

    param_groups = list(ensemble.experiment.parameter_configuration.keys())

    param_arrays = [
        _load_param_ensemble_array(ensemble, param_group, iens_active_index)
        for param_group in param_groups
    ]

    return np.vstack(param_arrays)


def _save_param_ensemble_array_to_disk(
    ensemble: Ensemble,
    param_ensemble_array: npt.NDArray[np.float64],
    param_group: str,
    iens_active_index: npt.NDArray[np.int_],
) -> None:
    config_node = ensemble.experiment.parameter_configuration[param_group]
    for i, realization in enumerate(iens_active_index):
        config_node.save_parameters(
            ensemble, int(realization), param_ensemble_array[:, i]
        )


def _load_param_ensemble_array(
    ensemble: Ensemble,
    param_group: str,
    iens_active_index: npt.NDArray[np.int_],
) -> npt.NDArray[np.float64]:
    config_node = ensemble.experiment.parameter_configuration[param_group]
    return config_node.load_parameters(ensemble, iens_active_index)


def _expand_wildcards(
    input_list: npt.NDArray[np.str_], patterns: list[str]
) -> list[str]:
    """
    Returns a sorted list of unique strings from `input_list` that match any
    of the specified wildcard patterns.

    Examples:
        >>> _expand_wildcards(np.array(["apple", "apricot", "banana"]), ["apricot", "apricot"])
        ['apricot']
        >>> _expand_wildcards(np.array(["apple", "banana", "apricot"]), [])
        []
        >>> _expand_wildcards(np.array(["dog", "deer", "frog"]), ["d*"])
        ['deer', 'dog']
        >>> _expand_wildcards(np.array(["apple", "APPLE", "Apple"]), ["apple"])
        ['apple']
    """  # noqa: E501
    matches = []
    for pattern in patterns:
        matches.extend([str(val) for val in input_list if fnmatch(val, pattern)])
    return sorted(set(matches))


class _OutlierColumns(StrEnum):
    response_std = "response_std"
    ens_mean = "response_mean"
    obs_scaling = "obs_error_scaling"
    scaled_std = "scaled_obs_error"


def _compute_observation_statuses(
    observations_and_responses: pl.DataFrame,
    global_std_scaling: float,
    outlier_settings: OutlierSettings,
    active_realizations: list[str],
) -> pl.DataFrame:
    """
    Computes and adds columns (named in _OutlierColumns) for:
     * response mean
     * response standard deviation
     * observation error scaling
     * scaled observation errors
     * status of (the responses of) each observation,
       corresponding to ObservationStatus
    """
    responses = observations_and_responses.select(active_realizations)
    response_stds = responses.with_columns(
        pl.concat_list("*").list.std(ddof=0).alias(_OutlierColumns.response_std)
    )[_OutlierColumns.response_std]

    with_outlier_info = (
        observations_and_responses.with_columns(
            responses.mean_horizontal().alias(_OutlierColumns.ens_mean),
            response_stds,
            # Inflating measurement errors by a factor sqrt(global_std_scaling) as shown
            # in for example evensen2018 - Analysis of iterative ensemble smoothers for
            # solving inverse problems.
            # `global_std_scaling` is 1.0 for ES.
            pl.lit(np.sqrt(global_std_scaling), dtype=pl.Float64).alias(
                _OutlierColumns.obs_scaling
            ),
        )
        .with_columns(
            (pl.col("std") * pl.col(_OutlierColumns.obs_scaling)).alias(
                _OutlierColumns.scaled_std
            )
        )
        .with_columns(
            pl.when(
                pl.any_horizontal(
                    [
                        pl.col(c).is_nan() | pl.col(c).is_null()
                        for c in active_realizations
                    ]
                )
            )
            .then(pl.lit(ObservationStatus.MISSING_RESPONSE))
            .when(pl.col(_OutlierColumns.response_std) <= outlier_settings.std_cutoff)
            .then(pl.lit(ObservationStatus.STD_CUTOFF))
            .when(
                abs(pl.col("observations") - pl.col(_OutlierColumns.ens_mean))
                > outlier_settings.alpha
                * (
                    pl.col(_OutlierColumns.response_std)
                    + pl.col(_OutlierColumns.scaled_std)
                )
            )
            .then(pl.lit(ObservationStatus.OUTLIER))
            .otherwise(pl.lit(ObservationStatus.ACTIVE))
            .alias("status")
        )
    )

    return with_outlier_info


def _auto_scale_observations(
    observations_and_responses: pl.DataFrame,
    auto_scale_observations: list[ObservationGroups],
    obs_mask: npt.NDArray[np.bool_],
    active_realizations: list[str],
    progress_callback: Callable[[AnalysisEvent], None],
) -> tuple[npt.NDArray[np.float64], pl.DataFrame] | tuple[None, None]:
    """
    Performs 'Auto Scaling' to mitigate issues with correlated observations,
    and saves computed scaling factors across input groups to ERT storage.
    """
    scaling_factors_dfs = []

    scaling_factors_updated = (
        observations_and_responses[_OutlierColumns.obs_scaling].to_numpy().copy()
    )

    obs_keys = observations_and_responses["observation_key"].to_numpy().astype(str)
    for input_group in auto_scale_observations:
        group = _expand_wildcards(obs_keys, input_group)
        obs_group_mask = np.isin(obs_keys, group) & obs_mask

        if not any(obs_group_mask):
            logger.error(f"No observations active for group: {input_group}")
            continue

        logger.info(f"Scaling observation group: {group}")

        data_for_obs = observations_and_responses.filter(obs_group_mask)
        scaling_factors, clusters, nr_components = misfit_preprocessor.main(
            data_for_obs.select(active_realizations).to_numpy(),
            data_for_obs.select(_OutlierColumns.scaled_std).to_numpy(),
        )

        scaling_factors_updated[obs_group_mask] *= scaling_factors

        scaling_factors_dfs.append(
            pl.DataFrame(
                {
                    "input_group": [", ".join(input_group)] * len(scaling_factors),
                    "index": data_for_obs["index"],
                    "obs_key": data_for_obs["observation_key"],
                    "scaling_factor": pl.Series(scaling_factors, dtype=pl.Float32),
                }
            )
        )

        progress_callback(
            AnalysisDataEvent(
                name="Auto scale: " + ", ".join(input_group),
                data=DataSection(
                    header=[
                        "Observation",
                        "Index",
                        "Cluster",
                        "Nr components",
                        "Scaling factor",
                    ],
                    data=np.array(
                        (
                            data_for_obs["observation_key"].to_numpy(),
                            data_for_obs["index"],
                            clusters,
                            nr_components.astype(int),
                            scaling_factors,
                        )
                    ).T,  # type: ignore
                ),
            )
        )

    if scaling_factors_dfs:
        return scaling_factors_updated, pl.concat(scaling_factors_dfs)
    else:
        msg = (
            "WARNING: Could not auto-scale the "
            f"observations {auto_scale_observations}. "
            f"No match with existing active observations {obs_keys}"
        )
        logger.warning(msg)
        print(msg)
        return None, None


def _preprocess_observations_and_responses(
    ensemble: Ensemble,
    observation_settings: ObservationSettings,
    global_std_scaling: float,
    iens_active_index: npt.NDArray[np.int_],
    selected_observations: Iterable[str],
    progress_callback: Callable[[AnalysisEvent], None],
) -> pl.DataFrame:
    observations_and_responses = ensemble.get_observations_and_responses(
        selected_observations,
        iens_active_index,
    )

    observations_and_responses = observations_and_responses.sort(
        by=["observation_key", "index"]
    )

    realization_responses = [str(i) for i in iens_active_index]

    observations_and_responses = _compute_observation_statuses(
        observations_and_responses,
        global_std_scaling,
        observation_settings.outlier_settings,
        realization_responses,
    )

    obs_mask = (
        observations_and_responses.select(pl.col("status") == ObservationStatus.ACTIVE)
        .to_numpy()
        .flatten()
    )

    if observation_settings.auto_scale_observations:
        updated_std_scales, scaling_factors_df = _auto_scale_observations(
            observations_and_responses,
            observation_settings.auto_scale_observations,
            obs_mask,
            realization_responses,
            progress_callback,
        )

        if updated_std_scales is not None and scaling_factors_df is not None:
            ensemble.save_observation_scaling_factors(scaling_factors_df)

            # Recompute with updated scales
            observations_and_responses = observations_and_responses.with_columns(
                pl.Series(updated_std_scales).alias(_OutlierColumns.obs_scaling)
            ).with_columns(
                (pl.col(_OutlierColumns.obs_scaling) * pl.col("std")).alias(
                    _OutlierColumns.scaled_std
                )
            )

    missing_observations = (
        observations_and_responses.filter(pl.col("status") != ObservationStatus.ACTIVE)[
            "observation_key"
        ]
        .unique()
        .to_list()
    )
    missing_observations.sort()

    if len(missing_observations) > 0:
        logger.warning(f"Deactivating observations: {missing_observations}")

    return observations_and_responses


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


def _copy_unupdated_parameters(
    all_parameter_groups: Iterable[str],
    updated_parameter_groups: Iterable[str],
    iens_active_index: npt.NDArray[np.int_],
    source_ensemble: Ensemble,
    target_ensemble: Ensemble,
) -> None:
    """
    Copies parameter groups that have not been updated from a source ensemble to a
    target ensemble. This function ensures that all realizations in the target ensemble
    have a complete set of parameters, including those that were not updated.
    This is necessary because users can choose not to update parameters but may still
    want to analyse them.

    Parameters:
        all_parameter_groups (list[str]): A list of all parameter groups.
        updated_parameter_groups (list[str]): A list of parameter groups that have
            already been updated.
        iens_active_index (npt.NDArray[np.int_]): An array of indices for the active
            realizations in the target ensemble.
        source_ensemble (Ensemble): The file system of the source ensemble, from which
            parameters are copied.
        target_ensemble (Ensemble): The file system of the target ensemble, to which
            parameters are saved.

    Returns:
        None: The function does not return any value but updates the target file system
            by copying over the parameters.
    """
    # Identify parameter groups that have not been updated
    not_updated_parameter_groups = list(
        set(all_parameter_groups) - set(updated_parameter_groups)
    )

    # Copy the non-updated parameter groups from source to target
    # for each active realization
    for parameter_group in not_updated_parameter_groups:
        for realization in iens_active_index:
            ds = source_ensemble.load_parameters(parameter_group, realization)
            target_ensemble.save_parameters(parameter_group, realization, ds)


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
        source_ensemble,
        observation_settings,
        global_scaling,
        iens_active_index,
        observations,
        progress_callback,
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
        param_ensemble_array = _load_param_ensemble_array(
            source_ensemble, param_group, iens_active_index
        )
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
                X_local = param_ensemble_array[param_batch_idx, :]
                if isinstance(config_node, GenKwConfig):
                    correlation_batch_callback = functools.partial(
                        correlation_callback,
                        cross_correlations_accumulator=cross_correlations,
                    )
                else:
                    correlation_batch_callback = None
                param_ensemble_array[param_batch_idx, :] = (
                    smoother_adaptive_es.assimilate(
                        X=X_local,
                        Y=S,
                        D=D,
                        # The user is responsible for scaling observation covariance
                        # (esmda usage):
                        alpha=1.0,
                        correlation_threshold=module.correlation_threshold,
                        cov_YY=cov_YY,
                        progress_callback=adaptive_localization_progress_callback,
                        correlation_callback=correlation_batch_callback,
                    )
                )

            if cross_correlations:
                assert isinstance(config_node, GenKwConfig)
                parameter_names = [
                    t["name"]  # type: ignore
                    for t in config_node.transform_function_definitions
                ]
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
            param_ensemble_array = param_ensemble_array @ T.astype(  # noqa: PLR6104
                param_ensemble_array.dtype
            )

        log_msg = f"Storing data for {param_group}.."
        logger.info(log_msg)
        progress_callback(AnalysisStatusEvent(msg=log_msg))
        start = time.time()

        _save_param_ensemble_array_to_disk(
            target_ensemble, param_ensemble_array, param_group, iens_active_index
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
