"""Adaptive localization update strategy."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING

import humanize
import numpy as np
import psutil
from iterative_ensemble_smoother import AdaptiveESMDA

from ert.analysis.event import AnalysisEvent, AnalysisMatrixEvent, AnalysisStatusEvent

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.config import ParameterConfig

    from ._protocol import ObservationContext

logger = logging.getLogger(__name__)

# When running adaptive localization we reserve some resources for
# the GUI and other applications that might be running
RESERVED_CPU_CORES = 2
NUM_JOBS_ADAPTIVE_LOC = max(
    1, ((psutil.cpu_count(logical=False) or 1) - RESERVED_CPU_CORES)
)


def _split_by_batchsize(
    arr: npt.NDArray[np.int_], batch_size: int
) -> list[npt.NDArray[np.int_]]:
    """Split an array into sub-arrays of a specified batch size.

    Parameters
    ----------
    arr : npt.NDArray[np.int_]
        Array of indices to split.
    batch_size : int
        Target size for each batch.

    Returns
    -------
    list[npt.NDArray[np.int_]]
        List of sub-arrays.

    Examples
    --------
    >>> num_params = 10
    >>> batch_size = 3
    >>> s = np.arange(0, num_params)
    >>> _split_by_batchsize(s, batch_size)
    [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
    """
    sections = 1 if batch_size > len(arr) else len(arr) // batch_size
    return np.array_split(arr, sections)


def _calculate_adaptive_batch_size(num_params: int, num_obs: int) -> int:
    """Calculate adaptive batch size to optimize memory usage.

    Adaptive Localization calculates the cross-covariance between parameters
    and responses. Cross-covariance is a matrix with shape (num_params x num_obs)
    which may be larger than memory. This function calculates a batch size
    that can fit into the available memory with a safety margin.

    Parameters
    ----------
    num_params : int
        Number of parameters to update.
    num_obs : int
        Number of observations.

    Returns
    -------
    int
        Batch size that fits in available memory.
    """
    available_memory_in_bytes = psutil.virtual_memory().available
    memory_safety_factor = 0.8
    bytes_in_float32 = 4
    return min(
        int(
            np.floor(
                (available_memory_in_bytes * memory_safety_factor)
                / (num_obs * bytes_in_float32)
            )
        ),
        num_params,
    )


class AdaptiveLocalizationUpdate:
    """Adaptive localization using correlation thresholds.

    This strategy uses adaptive localization to reduce spurious correlations
    by applying correlation thresholds during the update. Parameters are
    processed in batches to manage memory usage.

    Parameters
    ----------
    correlation_threshold : Callable[[int], float]
        Function that takes ensemble size and returns the correlation threshold.
    enkf_truncation : float
        Singular value truncation threshold for the smoother.
    rng : np.random.Generator
        Random number generator for reproducibility.
    progress_callback : Callable[[AnalysisEvent], None]
        Callback to report progress events.

    Attributes
    ----------
    _smoother : AdaptiveESMDA | None
        The adaptive ESMDA smoother instance (set after prepare()).
    _num_obs : int
        Number of observations.
    """

    def __init__(
        self,
        correlation_threshold: Callable[[int], float],
        enkf_truncation: float,
        rng: np.random.Generator,
        progress_callback: Callable[[AnalysisEvent], None],
    ) -> None:
        self._correlation_threshold_fn = correlation_threshold
        self._enkf_truncation = enkf_truncation
        self._rng = rng
        self._progress_callback = progress_callback
        self._smoother: AdaptiveESMDA | None = None
        self._num_obs: int = 0
        self._ensemble_size: int = 0
        self._ensemble_id: str = ""

    def prepare(self, obs_context: ObservationContext) -> None:
        """Initialize smoother and pre-compute matrices from observation data.

        Parameters
        ----------
        obs_context : ObservationContext
            Preprocessed observation and response data.
        """
        self._smoother = AdaptiveESMDA(
            covariance=obs_context.observation_errors**2,
            observations=obs_context.observation_values,
            alpha=1,
            seed=self._rng,
        )

        self._smoother.prepare_assimilation(
            Y=obs_context.responses,
            truncation=self._enkf_truncation,
            overwrite=False,
        )

        self._num_obs = obs_context.num_observations
        self._ensemble_size = obs_context.ensemble_size

    def update(
        self,
        param_ensemble: npt.NDArray[np.floating],
        param_config: ParameterConfig,
        non_zero_variance_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.floating]:
        """Update parameters using adaptive localization with batching.

        Parameters
        ----------
        param_ensemble : npt.NDArray[np.floating]
            Parameter ensemble array.
        param_config : ParameterConfig
            Configuration for this parameter type.
        non_zero_variance_mask : npt.NDArray[np.bool_]
            Boolean mask for parameters with non-zero variance.

        Returns
        -------
        npt.NDArray[np.floating]
            Updated parameter ensemble array.

        Raises
        ------
        RuntimeError
            If prepare() was not called before update().
        """
        if self._smoother is None:
            raise RuntimeError("prepare() must be called before update()")

        num_params = param_ensemble.shape[0]
        batch_size = _calculate_adaptive_batch_size(num_params, self._num_obs)
        batches = _split_by_batchsize(np.arange(0, num_params), batch_size)
        num_batches = len(batches)

        batch_info = f" and {num_batches} batches" if num_batches > 1 else ""
        self._progress_callback(
            AnalysisStatusEvent(
                msg=f"Updating {param_config.name} ({param_config.type.upper()}) "
                f"using adaptive localization, "
                f"{num_params} parameters, "
                f"{self._num_obs} observations, "
                f"{self._ensemble_size} realizations"
                f"{batch_info}"
            )
        )

        threshold = self._correlation_threshold_fn(self._ensemble_size)

        corr_XY_batches: list[npt.NDArray[np.float64]] = []

        def correlation_callback(
            corr_XY: npt.NDArray[np.float64],
            observations_per_parameter: npt.NDArray[np.int_],
        ) -> npt.NDArray[np.bool_]:
            corr_XY_batches.append(corr_XY.copy())
            return np.abs(corr_XY) > threshold

        start_time = time.perf_counter()
        for param_batch_idx in batches:
            update_idx = param_batch_idx[non_zero_variance_mask[param_batch_idx]]
            X_local = param_ensemble[update_idx, :]

            param_ensemble[update_idx, :] = self._smoother.assimilate_batch(
                X=X_local,
                correlation_callback=correlation_callback,
                overwrite=True,
                n_jobs=NUM_JOBS_ADAPTIVE_LOC,
            )
        elapsed = time.perf_counter() - start_time

        if corr_XY_batches:
            corr_XY_matrix = np.concatenate(corr_XY_batches, axis=0)
            self._progress_callback(
                AnalysisMatrixEvent(
                    name=f"corr_XY_{param_config.name}",
                    ensemble_id=self._ensemble_id,
                    matrix=corr_XY_matrix,
                )
            )

        self._progress_callback(
            AnalysisStatusEvent(
                msg=f"Updated {param_config.name} ({param_config.type.upper()}) "
                f"in {humanize.precisedelta(timedelta(seconds=elapsed))}",
                detail=True,
            )
        )

        return param_ensemble
