from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, TextIO

import numpy as np
import polars as pl

from ert.config import ESSettings, Field, ObservationSettings, SurfaceConfig

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

    from ert.config import ParameterConfig
    from ert.storage import Ensemble

logger = logging.getLogger(__name__)


def _create_combined_ensemble_mask(
    ens_mask: npt.NDArray[np.bool_], active_realizations: list[bool] | None
) -> npt.NDArray[np.bool_]:
    if active_realizations is None:
        return ens_mask

    ens_mask_indices = set(np.flatnonzero(ens_mask))
    active_realizations_indices = set(np.flatnonzero(active_realizations))

    if len(ens_mask) >= len(active_realizations):
        ens_mask_indices &= active_realizations_indices

        new_mask = np.zeros_like(ens_mask, dtype=bool)
        if ens_mask_indices:
            new_mask[list(ens_mask_indices)] = True
    else:
        active_realizations_indices &= ens_mask_indices

        new_mask = np.zeros_like(active_realizations, dtype=bool)
        if active_realizations_indices:
            new_mask[list(active_realizations_indices)] = True

    return new_mask


def perform_ensemble_update(
    observations: Iterable[str],
    observation_settings: ObservationSettings,
    global_scaling: float,
    ens_mask: npt.NDArray[np.bool_],
    source_ensemble: Ensemble,
    target_ensemble: Ensemble,
    progress_callback: Callable[[AnalysisEvent], None],
    strategy_map: dict[str, UpdateStrategy],
) -> SmootherSnapshot:
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
    observations : Iterable[str]
        Names of observations to use.
    observation_settings : ObservationSettings
        Settings for observation handling.
    global_scaling : float
        Global scaling factor for observations.
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

    Returns
    -------
    SmootherSnapshot
        Snapshot containing observation/response data and metadata.
    """
    parameters = list(strategy_map.keys())
    iens_active_index = np.flatnonzero(ens_mask)

    progress_callback(AnalysisStatusEvent(msg="Loading observations and responses.."))

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

    # Upcast to float64: the iterative ensemble smoother requires matching
    # dtypes, and benefits from higher precision during matrix inversions.
    responses = (
        filtered_data.select([*map(str, iens_active_index)])
        .to_numpy(order="c")
        .astype(np.float64)
    )
    observation_values = filtered_data["observations"].to_numpy().astype(np.float64)
    observation_errors = (
        filtered_data[_OutlierColumns.scaled_std].to_numpy().astype(np.float64)
    )

    num_obs = len(observation_values)

    smoother_snapshot = SmootherSnapshot(
        source_ensemble_name=source_ensemble.name,
        target_ensemble_name=target_ensemble.name,
        alpha=observation_settings.outlier_settings.alpha,
        std_cutoff=observation_settings.outlier_settings.std_cutoff,
        global_scaling=global_scaling,
    )
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

    # Extract observation locations when location data is available
    observation_locations: ObservationLocations | None = None
    has_location = (
        filtered_data["east"].is_not_null() & filtered_data["north"].is_not_null()
    ).to_numpy()
    if has_location.any():
        observation_locations = ObservationLocations(
            xpos=filtered_data["east"].to_numpy()[has_location],
            ypos=filtered_data["north"].to_numpy()[has_location],
            main_range=filtered_data["radius"].to_numpy()[has_location],
            location_mask=has_location,
        )

    obs_context = ObservationContext(
        responses=responses,
        observation_values=observation_values,
        observation_errors=observation_errors,
        observation_locations=observation_locations,
    )

    # Prepare each unique strategy once (multiple params may share the same instance)
    for strategy in set(strategy_map.values()):
        if isinstance(strategy, AdaptiveLocalizationUpdate):
            strategy._ensemble_id = str(target_ensemble.id)
        strategy.prepare(obs_context)

    # Update each parameter group
    logger.info(
        f"Updating {len(parameters)} parameter groups "
        f"with {num_obs} observations and {len(iens_active_index)} realizations"
    )
    for param_group in parameters:
        param_cfg = source_ensemble.experiment.parameter_configuration[param_group]
        param_ensemble_array = source_ensemble.load_parameters_numpy(
            param_group, iens_active_index
        )

        # Calculate variance for each parameter
        param_variance = np.var(param_ensemble_array, axis=1)
        non_zero_variance_mask = ~np.isclose(param_variance, 0.0)

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
            param_ensemble_array,
            param_cfg,
            non_zero_variance_mask,
        )

        # Save updated parameters
        target_ensemble.save_parameters_numpy(
            param_ensemble_array, param_group, iens_active_index
        )

    _copy_unupdated_parameters(
        list(source_ensemble.experiment.parameter_configuration.keys()),
        parameters,
        iens_active_index,
        source_ensemble,
        target_ensemble,
    )

    return smoother_snapshot


def build_strategy_map(
    parameters: Iterable[str],
    param_configs: Mapping[str, ParameterConfig],
    enkf_truncation: float,
    distance_localization: bool = False,
    localization: bool = False,
    correlation_threshold: Callable[[int], float] | None = None,
    rng: np.random.Generator | None = None,
    progress_callback: Callable[[AnalysisEvent], None] | None = None,
) -> dict[str, UpdateStrategy]:
    """Build a mapping from parameter group names to update strategies.

    Creates the appropriate update strategy for each parameter group based on
    the provided settings (standard, adaptive localization, or distance
    localization).

    Parameters
    ----------
    parameters : Iterable[str]
        Names of parameter groups to update.
    param_configs : Mapping[str, ParameterConfig]
        Parameter configuration mapping from the experiment.
    enkf_truncation : float
        Singular value truncation threshold (0, 1].
    distance_localization : bool
        Whether to use distance-based localization for Field/Surface params.
    localization : bool
        Whether to use adaptive localization.
    correlation_threshold : Callable[[int], float] | None
        Function that takes ensemble size and returns the correlation
        threshold. Required when ``localization`` is True.
    rng : np.random.Generator | None
        Random number generator for reproducibility.
    progress_callback : Callable[[AnalysisEvent], None] | None
        Callback for reporting progress.
    Returns
    -------
    dict[str, UpdateStrategy]
        Mapping from parameter group names to update strategies.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not progress_callback:
        progress_callback = noop_progress_callback

    strategy_map: dict[str, UpdateStrategy] = {}

    if distance_localization:
        field_strategy = DistanceLocalizationUpdate(
            enkf_truncation, rng, Field, progress_callback
        )
        surface_strategy = DistanceLocalizationUpdate(
            enkf_truncation, rng, SurfaceConfig, progress_callback
        )
        standard_strategy = StandardESUpdate(
            enkf_truncation,
            rng,
            progress_callback,
        )

        for param_name in parameters:
            param_cfg = param_configs[param_name]
            if isinstance(param_cfg, Field):
                strategy_map[param_name] = field_strategy
            elif isinstance(param_cfg, SurfaceConfig):
                strategy_map[param_name] = surface_strategy
            else:
                strategy_map[param_name] = standard_strategy

    elif localization:
        if correlation_threshold is None:
            raise ValueError(
                "correlation_threshold is required when localization is enabled"
            )
        adaptive_strategy = AdaptiveLocalizationUpdate(
            correlation_threshold, enkf_truncation, rng, progress_callback
        )
        for param_name in parameters:
            strategy_map[param_name] = adaptive_strategy

    else:
        standard_strategy = StandardESUpdate(
            enkf_truncation,
            rng,
            progress_callback,
        )
        for param_name in parameters:
            strategy_map[param_name] = standard_strategy

    return strategy_map


def smoother_update(
    prior_storage: Ensemble,
    posterior_storage: Ensemble,
    observations: Iterable[str],
    update_settings: ObservationSettings,
    strategy_map: dict[str, UpdateStrategy] | None = None,
    progress_callback: Callable[[AnalysisEvent], None] | None = None,
    global_scaling: float = 1.0,
    active_realizations: list[bool] | None = None,
) -> SmootherSnapshot:
    if not progress_callback:
        progress_callback = noop_progress_callback

    if strategy_map is None:
        settings = ESSettings()
        experiment = prior_storage.experiment
        strategy_map = build_strategy_map(
            parameters=experiment.update_parameters,
            param_configs=experiment.parameter_configuration,
            enkf_truncation=settings.enkf_truncation,
            progress_callback=progress_callback,
        )

    ens_mask = prior_storage.get_realization_mask_with_responses()
    ens_mask = _create_combined_ensemble_mask(ens_mask, active_realizations)

    smoother_snapshot: SmootherSnapshot | None = None
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
            smoother_snapshot = perform_ensemble_update(
                observations,
                update_settings,
                global_scaling,
                ens_mask,
                prior_storage,
                posterior_storage,
                progress_callback,
                strategy_map,
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
            ensemble_id=str(posterior_storage.id),
        )
    )
    return smoother_snapshot
