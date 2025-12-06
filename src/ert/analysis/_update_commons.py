from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from enum import StrEnum
from fnmatch import fnmatch

import numpy as np
import polars as pl
from numpy import typing as npt

from ert.config import ObservationGroups, OutlierSettings
from ert.storage import Ensemble

from . import misfit_preprocessor
from .event import (
    AnalysisDataEvent,
    AnalysisEvent,
    DataSection,
)
from .snapshots import (
    ObservationStatus,
)

logger = logging.getLogger(__name__)


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
    complete_df: pl.DataFrame | None = None
    for parameter_group in not_updated_parameter_groups:
        data = source_ensemble.load_parameters(parameter_group, iens_active_index)
        if isinstance(data, pl.DataFrame):
            if complete_df is None:
                complete_df = data
            else:
                complete_df = complete_df.join(data, on="realization")
        else:
            target_ensemble.save_parameters(dataset=data)

    if complete_df is not None:
        target_ensemble.save_parameters(complete_df)


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
    iens_active_index: npt.NDArray[np.int_],
    global_std_scaling: float,
    selected_observations: Iterable[str],
    outlier_settings: OutlierSettings | None = None,
    auto_scale_observations: list[ObservationGroups] | None = None,
    progress_callback: Callable[[AnalysisEvent], None] | None = None,
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
        observations_and_responses=observations_and_responses,
        active_realizations=realization_responses,
        global_std_scaling=global_std_scaling,
        outlier_settings=outlier_settings,
    )

    obs_mask = (
        observations_and_responses.select(pl.col("status") == ObservationStatus.ACTIVE)
        .to_numpy()
        .flatten()
    )

    if auto_scale_observations:
        assert progress_callback is not None
        updated_std_scales, scaling_factors_df = _auto_scale_observations(
            observations_and_responses,
            auto_scale_observations,
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


def _compute_observation_statuses(
    observations_and_responses: pl.DataFrame,
    active_realizations: list[str],
    global_std_scaling: float,
    outlier_settings: OutlierSettings | None = None,
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

    df_with_status = observations_and_responses

    obs_has_null_response_ = pl.any_horizontal(
        [pl.col(c).is_nan() | pl.col(c).is_null() for c in active_realizations]
    )

    if outlier_settings is None:
        return df_with_status.with_columns(
            pl.when(obs_has_null_response_)
            .then(pl.lit(ObservationStatus.MISSING_RESPONSE))
            .otherwise(pl.lit(ObservationStatus.ACTIVE))
            .alias("status")
        )

    assert global_std_scaling is not None

    responses = observations_and_responses.select(active_realizations)
    response_stds = responses.with_columns(
        pl.concat_list("*").list.std(ddof=0).alias(_OutlierColumns.response_std)
    )[_OutlierColumns.response_std]

    df_with_status = df_with_status.with_columns(
        responses.mean_horizontal().alias(_OutlierColumns.ens_mean),
        response_stds,
        # Inflating measurement errors by a factor sqrt(global_std_scaling) as shown
        # in for example evensen2018 - Analysis of iterative ensemble smoothers for
        # solving inverse problems.
        # `global_std_scaling` is 1.0 for ES.
        pl.lit(np.sqrt(global_std_scaling), dtype=pl.Float64).alias(
            _OutlierColumns.obs_scaling
        ),
    ).with_columns(
        (pl.col("std") * pl.col(_OutlierColumns.obs_scaling)).alias(
            _OutlierColumns.scaled_std
        )
    )

    df_with_status = df_with_status.with_columns(
        pl.when(obs_has_null_response_)
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

    return df_with_status


class _OutlierColumns(StrEnum):
    response_std = "response_std"
    ens_mean = "response_mean"
    obs_scaling = "obs_error_scaling"
    scaled_std = "scaled_obs_error"


def _all_parameters(
    ensemble: Ensemble,
    iens_active_index: npt.NDArray[np.int_],
) -> npt.NDArray[np.float64]:
    """Return all parameters in assimilation problem"""

    groups_to_update = [
        k for k, v in ensemble.experiment.parameter_configuration.items() if v.update
    ]
    param_arrays = [
        ensemble.load_parameters_numpy(param_group, iens_active_index)
        for param_group in groups_to_update
    ]

    return np.vstack(param_arrays)


class ErtAnalysisError(Exception):
    pass


def noop_progress_callback(_: AnalysisEvent) -> None:
    pass


def define_active_matrix_for_initial_ensemble(
    X_matrix: npt.NDArray[np.float64],
    X_active: npt.NDArray[np.bool] | None = None,
    min_real: int = 10,
) -> tuple[npt.NDArray[np.bool], npt.NDArray[np.bool], int]:
    """Create the X_active matrix or update it if it already exists.
    Uses two criteria to deactivate field parameters (mark them as not updatable)
    Criteria 1: If ensemble standard devation is 0 (all realizations are equal),
    the field parameter can not be updated and must be deactivated.
    Criteria 2: If a field parameter has both active and inactive realizations
    set the field parameter inactive if number of realizations is below a minimum.

    Args:
        X_matrix -  Ensemble of field parameters. Shape = (nparam, nreal)
        X_active -  Same size as X_matrix and contains True for field parameters
                    and realizations where the field parameter is active
                    and False if not active.
        min_real -  Minimum number of active realizations for the field parameter to
                    be defined as updatable.
    Returns:
        X_active matrix - created if input X_active is None
                    or updated if X_active input exists.
        updatable_field_params - Bool array with True for active field parameter
                    and False for inactive.
        number_of updatable_field_params  - Number of active parameters

    """
    assert X_matrix is not None
    nparam = X_matrix.shape[0]
    nreal = X_matrix.shape[1]

    if X_active is None:
        # Initialized to active for all field parameters for all realizations
        X_active = np.full((nparam, nreal), True, dtype=np.bool)

    print(f"{X_active.shape=}")
    # Check critera 1: Has some field parameters 0 ensemble std?
    X_std = X_matrix.std(axis=1)
    non_updatable = X_std == 0

    if np.sum(non_updatable) > 0:
        # Some field parameters have 0 variance
        X_active[non_updatable, :] = False

    # Check criteria 2: Has sufficient number of realizations?
    params_with_too_few_realizations = X_active.sum(axis=1) < min_real
    X_active[params_with_too_few_realizations, :] = False
    updatable_field_params = np.sum(X_active, axis=1) >= min_real
    number_of_updatable_params = np.sum(updatable_field_params)
    return X_active, updatable_field_params, number_of_updatable_params


def adjust_inactive_field_values_to_match_average_of_active_field_values(
    X_matrix: npt.NDArray[np.float64],
    X_active: npt.NDArray[np.bool],
) -> npt.NDArray[np.float64]:
    """This function will modify the ensemble of field parameters in X_matrix
    and return a modified version. The purpose is to ensure that field parameter
    realizations that are inactive don't contribute to the ensemble average
    which is taken over all field parameter realizations (both active and inactive)
    in the DistanceESMDA assimilate algorithm. This is done by calculating
    the ensemble average over the active realizations and
    assign the average value to the field parameter realizations defined as inactive.


    The steps in the algorithm is then:
    - Calculate average over the ensemble of the active (used) field parameter
      values for each individual field parameter.
    - For each field parameter having some active realizations, assign the
      ensemble average to the realizations that are defined as inactive.
    - For field parameters that are inactive for all realizations, do nothing.


    Args:
        X_matrix - Input ensemble of field parameters, shape = (nparam, nreal)
        X_active - Input matrix defining which field parameter is active
                   or inactive for each realization, shape = (nparam, nreal)
    Returns:
        Modified X_matrix

    """
    assert X_matrix is not None
    assert X_active is not None

    inactive = ~X_active
    # Safely compute the ensemble mean (row-wise mean) for active entries only
    active_real_per_field_param = np.sum(X_active, axis=1, keepdims=True)

    # Avoid division by zero by setting mean to 0
    # if there are no active realizations for a field parameter
    active_real_per_field_param[active_real_per_field_param == 0] = 1
    X_mean = np.zeros(X_matrix.shape, dtype=np.float64)
    X_mean[:, :] = (
        np.sum(X_matrix * X_active, axis=1, keepdims=True) / active_real_per_field_param
    )

    # Replace field parameters with some inactive realizations
    # with the average value over the active realizations.
    # Field parameters that are inactive are set to 0
    X_matrix[inactive] = X_mean[inactive]

    # Return the modified matrix
    return X_matrix
