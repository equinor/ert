from __future__ import annotations

import logging
import re
import time
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, TextIO

import numpy as np
import polars as pl

from ert.config import Field, ObservationSettings, SurfaceConfig

from ._update_commons import (
    ErtAnalysisError,
    _copy_unupdated_parameters,
    _OutlierColumns,
    _preprocess_observations_and_responses,
    noop_progress_callback,
)
from ._update_strategies import (
    AdaptiveLocalizationUpdate,
    DistanceLocalizationUpdate,
    ObservationContext,
    ObservationLocations,
    StandardESUpdate,
    UpdateStrategy,
)
from .event import (
    AnalysisCompleteEvent,
    AnalysisErrorEvent,
    AnalysisEvent,
    AnalysisStatusEvent,
    DataSection,
)
from .snapshots import ObservationStatus, SmootherSnapshot

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.config import ESSettings
    from ert.storage import Ensemble

logger = logging.getLogger(__name__)


def perform_ensemble_update(
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
    strategy_map: dict[str, UpdateStrategy],
) -> None:
    """
    Orchestrate ensemble-based parameter updates using configurable strategies.

    This function coordinates the parameter update process using a strategy pattern
    that supports multiple analysis algorithms (ES, IES, etc.). The workflow:
    1. Preprocessing observations and responses
    2. Preparing strategies with context data
    3. Delegating parameter updates to the mapped strategies
    4. Saving updated parameters to the target ensemble

    Parameters
    ----------
    parameters : Iterable[str]
        Names of parameter groups to update.
    observations : Iterable[str]
        Names of observations to use.
    rng : np.random.Generator
        Random number generator for reproducibility.
    module : ESSettings
        ES settings controlling update behavior.
    observation_settings : ObservationSettings
        Settings for observation handling.
    global_scaling : float
        Global scaling factor for observations.
    smoother_snapshot : SmootherSnapshot
        Snapshot object for storing results.
    ens_mask : npt.NDArray[np.bool_]
        Boolean mask for active realizations.
    source_ensemble : Ensemble
        Source ensemble to read parameters from.
    target_ensemble : Ensemble
        Target ensemble to save updated parameters to.
    progress_callback : Callable[[AnalysisEvent], None]
        Callback for reporting progress.
    strategy_map : dict[str, UpdateStrategy]
        Mapping from parameter group names to update strategies.
    """
    iens_active_index = np.flatnonzero(ens_mask)

    # Preprocess observations and responses
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

    responses = filtered_data.select([*map(str, iens_active_index)]).to_numpy(order="c")
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

    # Extract observation locations if distance localization is enabled
    observation_locations: ObservationLocations | None = None
    if module.distance_localization:
        has_location = (
            filtered_data["east"].is_not_null() & filtered_data["north"].is_not_null()
        ).to_numpy()
        observation_locations = ObservationLocations(
            xpos=filtered_data["east"].to_numpy()[has_location],
            ypos=filtered_data["north"].to_numpy()[has_location],
            main_range=filtered_data["radius"].to_numpy()[has_location],
            responses_with_loc=responses[has_location, :],
            observation_values=observation_values[has_location],
            observation_errors=observation_errors[has_location],
        )

    # Create observation context (minimal data container)
    obs_context = ObservationContext(
        responses=responses,
        observation_values=observation_values,
        observation_errors=observation_errors,
        observation_locations=observation_locations,
    )

    # Prepare each unique strategy once (multiple params may share the same instance)
    for strategy in set(strategy_map.values()):
        strategy.prepare(obs_context)

    # Update each parameter group
    for param_group in parameters:
        param_cfg = source_ensemble.experiment.parameter_configuration[param_group]
        param_ensemble_array = source_ensemble.load_parameters_numpy(
            param_group, iens_active_index
        )

        # Calculate variance for each parameter
        param_variance = np.var(param_ensemble_array, axis=1)
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

        # Get appropriate strategy for this parameter type
        strategy = strategy_map[param_group]

        # Delegate update to strategy
        param_ensemble_array = strategy.update(
            param_group,
            param_ensemble_array,
            param_cfg,
            non_zero_variance_mask,
        )

        # Save updated parameters
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

    # Create strategies based on settings and parameter types
    param_configs = prior_storage.experiment.parameter_configuration
    strategy_map: dict[str, UpdateStrategy] = {}

    if es_settings.distance_localization:
        # Distance localization: Field/Surface use distance strategy,
        # others use standard ES
        field_strategy = DistanceLocalizationUpdate(rng, Field)
        surface_strategy = DistanceLocalizationUpdate(rng, SurfaceConfig)
        standard_strategy = StandardESUpdate(
            smoother_snapshot, es_settings, rng, progress_callback
        )

        for param_name in parameters:
            param_cfg = param_configs[param_name]
            if isinstance(param_cfg, Field):
                strategy_map[param_name] = field_strategy
            elif isinstance(param_cfg, SurfaceConfig):
                strategy_map[param_name] = surface_strategy
            else:
                strategy_map[param_name] = standard_strategy

    elif es_settings.localization:
        # Adaptive localization for all parameters
        adaptive_strategy = AdaptiveLocalizationUpdate(
            es_settings, rng, progress_callback
        )
        for param_name in parameters:
            strategy_map[param_name] = adaptive_strategy

    else:
        # Standard ES for all parameters
        standard_strategy = StandardESUpdate(
            smoother_snapshot, es_settings, rng, progress_callback
        )
        for param_name in parameters:
            strategy_map[param_name] = standard_strategy

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
            perform_ensemble_update(
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
                strategy_map,
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
