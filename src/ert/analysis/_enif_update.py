from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import networkx as nx
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


def prune_nan_nodes(
    graph: nx.Graph[int], nan_mask: npt.NDArray[np.bool_]
) -> nx.Graph[int]:
    """Remove NaN-flagged nodes from a parameter graph and relabel to 0..n-1.

    After removal, the remaining nodes are relabeled to contiguous integers
    so that node k corresponds to column k in the NaN-filtered parameter array.
    Spatial adjacency between surviving nodes is preserved; no new edges are added.
    """
    nan_nodes = set(np.where(nan_mask)[0].tolist())
    graph = graph.copy()
    graph.remove_nodes_from(nan_nodes)
    return nx.convert_node_labels_to_integers(  # type: ignore[call-overload]
        graph, first_label=0, ordering="sorted"
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
        data = None
        if isinstance(e, ErtAnalysisError):
            data = e.data
        progress_callback(AnalysisErrorEvent(error_msg=str(e), data=data))
        raise

    progress_callback(
        AnalysisCompleteEvent(
            data=DataSection(
                header=smoother_snapshot.header,
                data=smoother_snapshot.csv,
                extra=smoother_snapshot.extra,
            ),
            posterior_id=str(posterior_storage.id),
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

    num_obs = len(observation_values)

    smoother_snapshot.observations_and_responses = preprocessed_data.drop(
        [*map(str, iens_active_index), "response_key"]
    ).select(
        "observation_key",
        "index",
        "observations",
        "std",
        "status",
        "missing_realizations",
    )

    if num_obs == 0:
        msg = "No active observations for update step"
        raise ErtAnalysisError(
            msg,
            data=DataSection(
                header=smoother_snapshot.header,
                data=smoother_snapshot.csv,
                extra=smoother_snapshot.extra,
            ),
        )

    # EnIF ###
    start_enif = time.time()

    updated_parameters = [
        p
        for p in parameters
        if source_ensemble.experiment.parameter_configuration[p].update
    ]

    # Load each parameter group once and reuse throughout
    param_arrays = {
        group: source_ensemble.load_parameters_numpy(group, iens_active_index)
        for group in updated_parameters
    }

    X_full = np.vstack(list(param_arrays.values()))

    nan_row_mask = np.any(np.isnan(X_full), axis=1)
    if nan_row_mask.any():
        num_nan = int(nan_row_mask.sum())
        num_all_nan = int(np.all(np.isnan(X_full), axis=1).sum())
        num_partial_nan = num_nan - num_all_nan
        log_msg = (
            f"EnIF: Excluding {num_nan}/{len(nan_row_mask)} parameter rows "
            f"containing NaN ({num_all_nan} fully inactive"
        )
        if num_partial_nan > 0:
            log_msg += (
                f", {num_partial_nan} partially active — "
                f"these will not be updated for any realization"
            )
        log_msg += ")"
        logger.info(log_msg)
        progress_callback(AnalysisStatusEvent(msg=log_msg))

    X_clean = X_full[~nan_row_mask]
    if X_clean.shape[0] == 0:
        msg = "All parameter rows contain NaN — cannot run EnIF update"
        raise ErtAnalysisError(
            msg,
            data=DataSection(
                header=smoother_snapshot.header,
                data=smoother_snapshot.csv,
                extra=smoother_snapshot.extra,
            ),
        )

    X_clean_scaler = StandardScaler()
    X_clean_scaled = X_clean_scaler.fit_transform(X_clean.T)

    # Call fit: Learn sparse linear map only
    H = linear_boost_ic_regression(
        U=X_clean_scaled,
        Y=S.T,
        verbose_level=5,
    )

    # Learn the precision matrix block-sparse over parameter groups
    Prec_u = sp.sparse.csc_matrix((0, 0), dtype=float)
    for param_group in updated_parameters:
        config_node = source_ensemble.experiment.parameter_configuration[param_group]
        X_local = param_arrays[param_group]

        local_nan_mask = np.any(np.isnan(X_local), axis=1)
        X_local_clean = X_local[~local_nan_mask]

        if X_local_clean.shape[0] == 0:
            continue

        X_local_scaler = StandardScaler()
        X_scaled = X_local_scaler.fit_transform(X_local_clean.T)

        graph_u_sub = config_node.load_parameter_graph()
        if local_nan_mask.any():
            graph_u_sub = prune_nan_nodes(graph_u_sub, local_nan_mask)

        # This works for up to ~10^5 parameters
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

    # Using iterative=False because the non-iterative solvers
    # is more deterministic across macos / linux than the
    # iterative solver, which makes testing easier.
    # However, the iterative solver probably uses less memory,
    # so this could be revisited in the future if memory usage becomes an issue.
    X_updated = gtmap.transport(
        X_clean_scaled,
        S.T,
        observation_values,
        update_indices=update_indices,
        iterative=False,
        verbose_level=5,
        seed=random_seed,
    )
    X_updated = X_clean_scaler.inverse_transform(X_updated).T

    X_full[~nan_row_mask] = X_updated

    # Iterate over parameters to store the updated ensemble
    log_msg = f"Storing {len(updated_parameters)} updated parameter groups"
    logger.info(log_msg)
    progress_callback(AnalysisStatusEvent(msg=log_msg))
    parameters_updated = 0
    for param_group in updated_parameters:
        parameters_to_update = param_arrays[param_group].shape[0]
        param_group_indices = np.arange(
            parameters_updated, parameters_updated + parameters_to_update
        )
        target_ensemble.save_parameters_numpy(
            X_full[param_group_indices, :],
            param_group,
            iens_active_index,
        )
        parameters_updated += parameters_to_update

    _copy_unupdated_parameters(
        list(source_ensemble.experiment.parameter_configuration.keys()),
        updated_parameters,
        iens_active_index,
        source_ensemble,
        target_ensemble,
    )

    stop_enif = time.time()
    logger.info(f"EnIF total update time: {stop_enif - start_enif} seconds")
