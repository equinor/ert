from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Iterable

import numpy as np
import polars as pl
import scipy as sp
from graphite_maps.enif import EnIF  # type: ignore
from graphite_maps.linear_regression import linear_boost_ic_regression  # type: ignore
from graphite_maps.precision_estimation import (  # type: ignore
    fit_precision_cholesky_approximate,
)
from numpy import typing as npt
from sklearn.preprocessing import StandardScaler  # type: ignore

from ert.analysis._es_update import logger
from ert.storage import Ensemble

from ._update_commons import (
    ErtAnalysisError,
    _all_parameters,
    _copy_unupdated_parameters,
    _preprocess_observations_and_responses,
    noop_progress_callback,
)
from .event import (
    AnalysisCompleteEvent,
    AnalysisErrorEvent,
    AnalysisEvent,
    AnalysisStatusEvent,
    DataSection,
)
from .snapshots import (
    ObservationStatus,
    SmootherSnapshot,
)


def enif_update(
    prior_storage: Ensemble,
    posterior_storage: Ensemble,
    observations: Iterable[str],
    parameters: Iterable[str],
    random_seed: int,
    progress_callback: Callable[[AnalysisEvent], None] | None = None,
) -> SmootherSnapshot:
    if not progress_callback:
        progress_callback = noop_progress_callback

    ens_mask = prior_storage.get_realization_mask_with_responses()

    smoother_snapshot = SmootherSnapshot(
        source_ensemble_name=prior_storage.name,
        target_ensemble_name=posterior_storage.name,
        alpha=-1,
        std_cutoff=-1,
        global_scaling=-1,
    )

    try:
        analysis_EnIF(
            parameters,
            observations,
            random_seed,
            smoother_snapshot,
            ens_mask,
            prior_storage,
            posterior_storage,
            progress_callback,
        )
    except Exception as e:
        traceback.print_tb(e.__traceback__)
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


def analysis_EnIF(
    parameters: Iterable[str],
    observations: Iterable[str],
    random_seed: int | None,
    smoother_snapshot: SmootherSnapshot,
    ens_mask: npt.NDArray[np.bool_],
    source_ensemble: Ensemble,
    target_ensemble: Ensemble,
    progress_callback: Callable[[AnalysisEvent], None],
    global_scaling: float = 1.0,
) -> None:
    iens_active_index = np.flatnonzero(ens_mask)

    progress_callback(AnalysisStatusEvent(msg="Loading observations and responses.."))
    preprocessed_data = _preprocess_observations_and_responses(
        source_ensemble,
        selected_observations=observations,
        iens_active_index=iens_active_index,
        global_std_scaling=global_scaling,
    )

    filtered_data = preprocessed_data.filter(
        pl.col("status") == ObservationStatus.ACTIVE
    )

    S = filtered_data.select([*map(str, iens_active_index)]).to_numpy(order="c")
    observation_values = filtered_data["observations"].to_numpy()
    observation_errors = filtered_data["std"].to_numpy()

    progress_callback(AnalysisStatusEvent(msg="Loading observations and responses.."))
    num_obs = len(observation_values)

    smoother_snapshot.observations_and_responses = preprocessed_data.drop(
        [*map(str, iens_active_index), "response_key"]
    ).select(
        "observation_key",
        "index",
        "observations",
        "std",
        "status",
    )

    if num_obs == 0:
        msg = "No active observations for update step"
        progress_callback(AnalysisErrorEvent(error_msg=msg, data=smoother_snapshot))
        raise ErtAnalysisError(msg)

    # EnIF ###
    start_enif = time.time()

    # Load all parameters at once
    X_full = _all_parameters(
        ensemble=source_ensemble,
        iens_active_index=iens_active_index,
    )

    X_full_scaler = StandardScaler()
    X_full_scaled = X_full_scaler.fit_transform(X_full.T)

    # Call fit: Learn sparse linear map only
    H = linear_boost_ic_regression(
        U=X_full_scaled,
        Y=S.T,
        verbose_level=5,
    )

    # Learn the precision matrix block-sparse over parameter groups
    Prec_u = sp.sparse.csc_matrix((0, 0), dtype=float)
    for param_group in parameters:
        config_node = source_ensemble.experiment.parameter_configuration[param_group]
        X_local = source_ensemble.load_parameters_numpy(param_group, iens_active_index)
        X_local_scaler = StandardScaler()
        X_scaled = X_local_scaler.fit_transform(X_local.T)

        graph_u_sub = config_node.load_parameter_graph()

        # This will work for dim(X_scaled) on order O(n^5)
        Prec_u_sub = fit_precision_cholesky_approximate(
            X_scaled,
            graph_u_sub,
            neighbourhood_expansion=2,
            verbose_level=2,
            use_tqdm=True,
        )

        # Add to block-diagonal full precision
        Prec_u = sp.sparse.block_diag((Prec_u, Prec_u_sub), format="csc")

    # Precision of observation errors
    Prec_eps = sp.sparse.diags(
        [1.0 / observation_errors**2],
        offsets=[0],
        shape=(num_obs, num_obs),
        format="csc",
    )

    # Initialize EnIF object with full precision matrices
    gtmap = EnIF(
        Prec_u=Prec_u,
        Prec_eps=Prec_eps,
        H=H,
    )

    update_indices = gtmap.get_update_indices(
        neighbor_propagation_order=15, verbose_level=1
    )

    X_full = gtmap.transport(
        X_full_scaled,
        S.T,
        observation_values,
        update_indices=update_indices,
        iterative=True,
        verbose_level=5,
        seed=random_seed,
    )
    X_full = X_full_scaler.inverse_transform(X_full).T

    # Iterate over parameters to store the updated ensemble
    parameters_updated = 0
    for param_group in parameters:
        log_msg = f"Storing data for {param_group}.."
        logger.info(log_msg)
        progress_callback(AnalysisStatusEvent(msg=log_msg))
        start = time.time()

        param_ensemble_array = source_ensemble.load_parameters_numpy(
            param_group, iens_active_index
        )
        parameters_to_update = param_ensemble_array.shape[0]
        param_group_indices = np.arange(
            parameters_updated, parameters_updated + parameters_to_update
        )
        target_ensemble.save_parameters_numpy(
            X_full[param_group_indices, :],
            param_group,
            iens_active_index,
        )
        parameters_updated += parameters_to_update

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

    stop_enif = time.time()
    logger.info(f"EnIF total update time: {stop_enif - start_enif} seconds")
