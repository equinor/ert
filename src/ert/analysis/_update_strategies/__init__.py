"""Update strategies for ensemble parameter updates.

This package provides a Strategy + Factory pattern for parameter updates,
allowing different update methods (standard ES, adaptive localization,
distance-based localization) to be applied based on parameter types
and configuration settings.

Example usage:
    from ert.analysis._update_strategies import (
        UpdateContext,
        UpdateStrategyFactory,
    )

    # Create factory with settings
    factory = UpdateStrategyFactory(es_settings, smoother_snapshot)

    # For distance localization, extract observation locations first
    if es_settings.distance_localization:
        factory.extract_observation_locations(filtered_data, responses)

    # Create strategies
    strategies = factory.create_strategies()

    # Update each parameter group
    for param_group in parameters:
        strategy = factory.get_strategy_for(param_config, strategies)
        param_array = strategy.update(
            param_group, param_array, param_config, mask, context
        )
"""

from ._adaptive import AdaptiveLocalizationUpdate
from ._distance import (
    DistanceLocalizationFieldUpdate,
    DistanceLocalizationSurfaceUpdate,
)
from ._factory import UpdateStrategyFactory
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
    "UpdateStrategyFactory",
]
