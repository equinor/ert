"""Update strategies for ensemble parameter updates.

This package provides different update strategies for parameter updates,
allowing different update methods (standard ES, adaptive localization,
distance-based localization) to be applied to different parameters.

Strategy Lifecycle:
    1. Create strategy instances with configuration
    2. Call strategy.prepare(context) to initialize with observation data
    3. Call strategy.update() for each parameter group

Example usage:
    from ert.analysis._update_strategies import (
        StandardESUpdate,
        AdaptiveLocalizationUpdate,
        UpdateContext,
    )

    # Create strategies
    standard_strategy = StandardESUpdate(smoother_snapshot)
    adaptive_strategy = AdaptiveLocalizationUpdate()

    # Build strategy map (parameter_name -> strategy)
    strategy_map = {
        "PORO": adaptive_strategy,
        "PERM": standard_strategy,
    }

    # Prepare strategies with context (called by perform_ensemble_update)
    for strategy in set(strategy_map.values()):
        strategy.prepare(context)

    # Update each parameter group
    for param_group, strategy in strategy_map.items():
        param_array = strategy.update(
            param_group, param_array, param_config, mask, context
        )
"""

from ._adaptive import AdaptiveLocalizationUpdate
from ._distance import (
    DistanceLocalizationFieldUpdate,
    DistanceLocalizationSurfaceUpdate,
)
from ._protocol import (
    ObservationLocations,
    TimedIterator,
    UpdateContext,
    UpdateStrategy,
)
from ._standard import StandardESUpdate

__all__ = [
    "AdaptiveLocalizationUpdate",
    "DistanceLocalizationFieldUpdate",
    "DistanceLocalizationSurfaceUpdate",
    "ObservationLocations",
    "StandardESUpdate",
    "TimedIterator",
    "UpdateContext",
    "UpdateStrategy",
]
