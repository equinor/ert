"""Update strategies for ensemble parameter updates.

This package provides different update strategies for parameter updates,
allowing different update methods (standard ES, adaptive localization,
distance-based localization) to be applied to different parameters.

Strategy Lifecycle:
    1. Create strategy instances with dependencies (rng, settings, callback)
    2. Call strategy.prepare(obs_context) to initialize with observation data
    3. Call strategy.update() for each parameter group

Example usage:
    from ert.analysis._update_strategies import (
        StandardESUpdate,
        AdaptiveLocalizationUpdate,
        ObservationContext,
    )

    # Create strategies with dependencies
    standard_strategy = StandardESUpdate(
        smoother_snapshot, settings, rng, progress_callback
    )
    adaptive_strategy = AdaptiveLocalizationUpdate(
        settings, rng, progress_callback
    )

    # Build strategy map (parameter_name -> strategy)
    strategy_map = {
        "PORO": adaptive_strategy,
        "PERM": standard_strategy,
    }

    # Create observation context from preprocessed data
    obs_context = ObservationContext(
        responses=responses,
        observation_values=obs_values,
        observation_errors=obs_errors,
    )

    # Prepare strategies (called by perform_ensemble_update)
    for strategy in set(strategy_map.values()):
        strategy.prepare(obs_context)

    # Update each parameter group
    for param_group, strategy in strategy_map.items():
        param_array = strategy.update(
            param_group, param_array, param_config, mask
        )
"""

from ._adaptive import AdaptiveLocalizationUpdate
from ._distance import DistanceLocalizationUpdate
from ._protocol import (
    ObservationContext,
    ObservationLocations,
    TimedIterator,
    UpdateStrategy,
)
from ._standard import StandardESUpdate

__all__ = [
    "AdaptiveLocalizationUpdate",
    "DistanceLocalizationUpdate",
    "ObservationContext",
    "ObservationLocations",
    "StandardESUpdate",
    "TimedIterator",
    "UpdateStrategy",
]
