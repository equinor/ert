"""Protocol and data classes for parameter update strategies."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, Self, TypeVar

import numpy as np
import numpy.typing as npt

from ert.analysis.event import AnalysisEvent, AnalysisTimeEvent

if TYPE_CHECKING:
    from ert.config import ParameterConfig


T = TypeVar("T")


class TimedIterator(Generic[T]):
    """Iterator wrapper that reports progress timing via callback.

    Wraps a Sequence and provides single-pass iteration with progress
    reporting. Also exposes __len__ and __getitem__ to structurally
    satisfy the Sequence protocol expected by iterative_ensemble_smoother's
    progress_callback type hint.

    Parameters
    ----------
    iterable : Sequence[T]
        The underlying sequence to iterate over.
    callback : Callable[[AnalysisEvent], None]
        Callback function to report progress events.

    Attributes
    ----------
    SEND_FREQUENCY : float
        Minimum seconds between progress updates (default 1.0).
    """

    SEND_FREQUENCY = 1.0  # seconds

    def __init__(
        self, iterable: Sequence[T], callback: Callable[[AnalysisEvent], None]
    ) -> None:
        self._start_time = time.perf_counter()
        self._iterable = iterable
        self._callback = callback
        self._index = 0
        self._last_send_time = 0.0

    def __len__(self) -> int:
        return len(self._iterable)

    def __getitem__(self, index: int) -> T:
        return self._iterable[index]

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        try:
            result = self._iterable[self._index]
        except IndexError as e:
            raise StopIteration from e

        if self._index != 0:
            elapsed_time = time.perf_counter() - self._start_time
            estimated_remaining_time = (elapsed_time / (self._index)) * (
                len(self._iterable) - self._index
            )
            if elapsed_time - self._last_send_time > self.SEND_FREQUENCY:
                self._callback(
                    AnalysisTimeEvent(
                        remaining_time=estimated_remaining_time,
                        elapsed_time=elapsed_time,
                    )
                )
                self._last_send_time = elapsed_time

        self._index += 1
        return result


@dataclass
class ObservationLocations:
    """Observation location data for distance-based localization methods.

    Contains coordinates and correlation ranges for observations that have
    spatial location information. All arrays are filtered to only include
    observations that have valid location data.
    """

    xpos: npt.NDArray[np.float64]
    """X coordinates of observations (easting)."""

    ypos: npt.NDArray[np.float64]
    """Y coordinates of observations (northing)."""

    main_range: npt.NDArray[np.float64]
    """Correlation range (radius) for each observation."""

    responses_with_loc: npt.NDArray[np.float64]
    """Response matrix filtered to observations with locations."""

    observation_values: npt.NDArray[np.float64]
    """Observation values filtered to observations with locations."""

    observation_errors: npt.NDArray[np.float64]
    """Scaled observation errors filtered to observations with locations."""


@dataclass(frozen=True)
class ObservationContext:
    """Preprocessed observation data for parameter updates.

    This is a minimal, immutable data container holding only the observation
    and response data computed during preprocessing. Runtime dependencies
    (rng, settings, progress_callback) are passed to strategies at construction.
    """

    responses: npt.NDArray[np.float64]
    """Response matrix (num_obs x ensemble_size)."""

    observation_values: npt.NDArray[np.float64]
    """Observation values."""

    observation_errors: npt.NDArray[np.float64]
    """Scaled observation errors (standard deviations)."""

    observation_locations: ObservationLocations | None = None
    """Observation locations for distance-based localization (optional)."""

    @property
    def ensemble_size(self) -> int:
        """Number of active realizations (inferred from responses shape)."""
        return self.responses.shape[1]

    @property
    def num_observations(self) -> int:
        """Number of observations."""
        return len(self.observation_values)


class UpdateStrategy(Protocol):
    """Protocol for parameter update strategies.

    Each strategy implements a specific algorithm for updating ensemble
    parameters based on observations. Strategies receive runtime dependencies
    (rng, settings, progress_callback) at construction, and observation data
    via prepare().

    Lifecycle:
        1. Create strategy with dependencies (rng, settings, progress_callback)
        2. Call prepare(obs_context) to initialize with observation data
        3. Call update() for each parameter group
    """

    def prepare(self, obs_context: ObservationContext) -> None:
        """Initialize the strategy with observation data.

        Called once before any update() calls. Performs any expensive
        pre-computation (e.g., computing transition matrices).

        Parameters
        ----------
        obs_context : ObservationContext
            Preprocessed observation and response data.
        """
        ...

    def update(
        self,
        param_group: str,
        param_ensemble: npt.NDArray[np.float64],
        param_config: ParameterConfig,
        non_zero_variance_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float64]:
        """Update parameters using this strategy's algorithm.

        Parameters
        ----------
        param_group : str
            Name of the parameter group.
        param_ensemble : npt.NDArray[np.float64]
            Parameter ensemble array (num_params x ensemble_size).
        param_config : ParameterConfig
            Configuration for this parameter type.
        non_zero_variance_mask : npt.NDArray[np.bool_]
            Boolean mask for parameters with non-zero variance.

        Returns
        -------
        npt.NDArray[np.float64]
            Updated parameter ensemble array.
        """
        ...
