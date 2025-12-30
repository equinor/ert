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


def adjust_inactive_field_values_to_match_average_of_active_field_values(
    X_matrix: npt.NDArray[np.float64],
    X_active: npt.NDArray[np.bool],
    min_real: int = 10,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool]]:
    """This function will modify the ensemble of field parameters in X_matrix
    and return a modified version. The purpose is to avoid the workaround step when
    generating the realizations of field parameters where field parameters in ertbox
    that is not used in the geomodel grid zone must be filled with some sensible values
    anyway since ERT requires that all grid cells in ertbox grid is assumed to contain
    sensible values for all realizations.

    The update algorithm in ERT should be implemented such that it removes
    all field parameters with 0 variance from the update and only keep the value
    unchanged.

    For field parameters in ertbox grid that is NOT USED IN ANY realizations
    by the geogrid, the field parameter values should be set to a constant
    to ensure the variance is 0 such that it is removed from being updated.
    For field parameters in ertbox grid that is NOT USED IN ALL realizations
    by the geogrid, but only for some realizations, the unused field parameters
    should be modified and set to the average value of those realizations
    where the field parameter is used In the geomodel grid. This ensures that the
    calculation of the ensemble mean in the update algorithm will give the value
    that is equal to the average over the realizations where it is used in the geomodel.
    This ensures that no infill values in the ertbox grid for the field parameter values
    will have any effect on the ensemble average.

    If there are grid cells with field parameters which is used for only very few
    realizations in the geomodel grid (e.g. <= 10), the estimated mean value has
    low confidence. In this case it can be better to avoid updating the field
    parameter. To avoid updating the field parameter in this case, ensure that
    it has 0 variance so that it is removed from the field parameters that
    is to be updated. Therefore, in this case assign the ensemble average value to
    all realizations, not only those that does not have this field parameter as
    active in the geomodel but all realizations.

    The steps in the algorithm is then:
    - Calculate average over the ensemble of the active (used) field parameter
      values for each individual field parameter.
    - For each grid cell (i,j,k) in ertbox, check number of active realizations
      and depending on the number (if it is larger than or less than min_real)
      either assign the ensemble mean value to all realizations where the field
      parameter is inactive or assign the ensemble mean value to all realizations.

    Args:
        X_matrix - Input ensemble of field parameters, shape = (nparam, nreal)
        X_active - Input matrix defining which field parameter is active
            or inactive for each realization, shape = (nparam, nreal)
        min_real - Integer which is a lower limit of number of realization
            where it is acceptable to do any update. If number of realizations
            with active field parameter in an ertbox grid cell (i,j,k)
            is less than min_real, the ensemble for this field parameter will
            be set to a constant value to ensure 0 variance so that it is
            not updated.
    Returns:
        Modified X_matrix - Modified to ensure that ensemble average is not
            affected by values correponding to inactive realizations and that
            field parameters with very few realizations are not updated by
            setting a constant value in all realizations to ensure 0 variance.
        Modified X_active matrix where field parameters with few realizations
            are defined as inactive (which means not updatable)

    """

    # Create the inactive mask
    inactive_matrix = ~X_active

    # Safely compute the row-wise mean for active entries only
    row_active_counts = np.sum(
        X_active, axis=1, keepdims=True
    )  # Count active entries per row

    # Avoid division by zero by setting invalid rows to 0 temporarily
    row_active_counts[row_active_counts == 0] = 1
    X_mean = np.zeros(X_matrix.shape, dtype=np.float64)
    X_mean[:, :] = (
        np.sum(X_matrix * X_active, axis=1, keepdims=True) / row_active_counts
    )

    # Replace inactive values with the row mean
    X_matrix_modified = X_matrix.copy()
    X_matrix_modified[inactive_matrix] = X_mean[inactive_matrix]

    # Count the number of active realizations per row
    number_of_active_realizations_per_field_param = X_active.sum(axis=1)

    # Handle rows with fewer than min_real active realizations
    rows_to_replace = number_of_active_realizations_per_field_param < min_real

    # For rows where number of active realizations is less than min_real
    # set the field values to a constant , e.g. 0
    X_matrix_modified[rows_to_replace, :] = 0
    X_active_modified = X_active.copy()
    X_active_modified[rows_to_replace, :] = False

    # Return the modified matrix
    return X_matrix_modified, X_active_modified
