"""Factory for creating update strategies based on settings and parameter types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from ._adaptive import AdaptiveLocalizationUpdate
from ._distance import (
    DistanceLocalizationFieldUpdate,
    DistanceLocalizationSurfaceUpdate,
)
from ._protocol import ObservationLocations, UpdateStrategy
from ._standard import StandardESUpdate

if TYPE_CHECKING:
    from ert.analysis.snapshots import SmootherSnapshot
    from ert.config import ESSettings, ParameterConfig


class UpdateStrategyFactory:
    """Factory for creating update strategies based on ES settings.

    The factory examines the ES settings and creates appropriate strategies
    for different parameter types. It handles:

    - Standard ES update (no localization)
    - Adaptive localization (correlation-based)
    - Distance-based localization (spatial correlation for Fields/Surfaces)

    When distance localization is enabled, GenKw parameters still use
    standard ES update.

    Parameters
    ----------
    settings : ESSettings
        ES analysis settings determining which strategies to use.
    smoother_snapshot : SmootherSnapshot
        Snapshot for error reporting in strategies.

    Attributes
    ----------
    _obs_locations : ObservationLocations | None
        Extracted observation locations for distance localization.
    """

    def __init__(
        self,
        settings: ESSettings,
        smoother_snapshot: SmootherSnapshot,
    ) -> None:
        self._settings = settings
        self._smoother_snapshot = smoother_snapshot
        self._obs_locations: ObservationLocations | None = None

    def extract_observation_locations(
        self,
        filtered_data: pl.DataFrame,
        responses: npt.NDArray[np.float64],
        observation_values: npt.NDArray[np.float64],
        observation_errors: npt.NDArray[np.float64],
    ) -> ObservationLocations | None:
        """Extract observation location data for distance-based localization.

        Parameters
        ----------
        filtered_data : pl.DataFrame
            Filtered observation/response dataframe.
        responses : npt.NDArray[np.float64]
            Full response matrix.
        observation_values : npt.NDArray[np.float64]
            Full observation values array.
        observation_errors : npt.NDArray[np.float64]
            Full scaled observation errors array.

        Returns
        -------
        ObservationLocations | None
            Observation locations if distance localization is enabled, else None.
        """

        if not self._settings.distance_localization:
            return None

        # Filter to observations with location data
        has_location = (
            filtered_data["east"].is_not_null() & filtered_data["north"].is_not_null()
        ).to_numpy()

        self._obs_locations = ObservationLocations(
            xpos=filtered_data["east"].to_numpy()[has_location],
            ypos=filtered_data["north"].to_numpy()[has_location],
            main_range=filtered_data["radius"].to_numpy()[has_location],
            responses_with_loc=responses[has_location, :],
            observation_values=observation_values[has_location],
            observation_errors=observation_errors[has_location],
        )

        return self._obs_locations

    def create_strategies(self) -> list[UpdateStrategy]:
        """Create all strategies needed based on current settings.

        Returns
        -------
        list[UpdateStrategy]
            List of strategy instances. The order matters - strategies
            are checked in order for can_handle().

        Raises
        ------
        RuntimeError
            If distance localization is enabled but observation locations
            were not extracted first.
        """
        strategies: list[UpdateStrategy] = []

        if self._settings.distance_localization:
            if self._obs_locations is None:
                raise RuntimeError(
                    "Must call extract_observation_locations() before "
                    "create_strategies() when distance_localization is enabled"
                )
            # Distance localization strategies for Field and Surface
            # GenKw uses standard update even with distance localization
            strategies.extend(
                [
                    DistanceLocalizationFieldUpdate(self._obs_locations),
                    DistanceLocalizationSurfaceUpdate(self._obs_locations),
                    StandardESUpdate(self._smoother_snapshot),
                ]
            )

        elif self._settings.localization:
            # Adaptive localization for all parameter types
            strategies.append(AdaptiveLocalizationUpdate())

        else:
            # Standard ES update without localization
            strategies.append(StandardESUpdate(self._smoother_snapshot))

        return strategies

    def get_strategy_for(
        self,
        param_config: ParameterConfig,
        strategies: list[UpdateStrategy],
    ) -> UpdateStrategy:
        """Find the appropriate strategy for a parameter type.

        Strategies are checked in order of the list. The first strategy
        that can handle the parameter type is returned.

        Parameters
        ----------
        param_config : ParameterConfig
            Configuration for the parameter to update.
        strategies : list[UpdateStrategy]
            List of available strategies.

        Returns
        -------
        UpdateStrategy
            The first strategy that can handle this parameter type.

        Raises
        ------
        ValueError
            If no strategy can handle the parameter type.
        """
        for strategy in strategies:
            if strategy.can_handle(param_config):
                return strategy

        raise ValueError(
            f"No strategy found for parameter type {type(param_config).__name__}. "
            f"Available strategies: {[type(s).__name__ for s in strategies]}"
        )
