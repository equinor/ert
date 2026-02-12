"""Adaptive localization update strategy."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import psutil
from iterative_ensemble_smoother.experimental import AdaptiveESMDA

from ert.analysis.event import AnalysisStatusEvent

from ._protocol import TimedIterator

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.config import ParameterConfig

    from ._protocol import UpdateContext

logger = logging.getLogger(__name__)

T = TypeVar("T")

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

    Attributes
    ----------
    _smoother : AdaptiveESMDA | None
        The adaptive ESMDA smoother instance.
    _cov_YY : npt.NDArray[np.float64] | None
        Pre-computed response covariance matrix.
    _D : npt.NDArray[np.float64] | None
        Perturbed observations matrix.
    _num_obs : int
        Number of observations.
    """

    def __init__(self) -> None:
        self._smoother: AdaptiveESMDA | None = None
        self._cov_YY: npt.NDArray[np.float64] | None = None
        self._D: npt.NDArray[np.float64] | None = None
        self._num_obs: int = 0

    def initialize(self, context: UpdateContext) -> None:
        """Initialize the adaptive smoother and pre-compute matrices.

        Parameters
        ----------
        context : UpdateContext
            Shared update context with observations and settings.
        """
        self._smoother = AdaptiveESMDA(
            covariance=context.observation_errors**2,
            observations=context.observation_values,
            seed=context.rng,
        )

        # Pre-calculate cov_YY for efficiency
        self._cov_YY = np.atleast_2d(np.cov(context.responses))

        # Perturb observations
        self._D = self._smoother.perturb_observations(
            ensemble_size=context.ensemble_size, alpha=1.0
        )

        self._num_obs = len(context.observation_values)

    def can_handle(self, param_config: ParameterConfig) -> bool:
        """Check if this strategy handles the parameter type.

        Parameters
        ----------
        param_config : ParameterConfig
            Configuration for the parameter.

        Returns
        -------
        bool
            Always True since adaptive localization handles all types.
        """
        return True

    def update(
        self,
        param_group: str,
        param_ensemble: npt.NDArray[np.float64],
        param_config: ParameterConfig,
        non_zero_variance_mask: npt.NDArray[np.bool_],
        context: UpdateContext,
    ) -> npt.NDArray[np.float64]:
        """Update parameters using adaptive localization with batching.

        Parameters
        ----------
        param_group : str
            Name of the parameter group.
        param_ensemble : npt.NDArray[np.float64]
            Parameter ensemble array.
        param_config : ParameterConfig
            Configuration for this parameter type.
        non_zero_variance_mask : npt.NDArray[np.bool_]
            Boolean mask for parameters with non-zero variance.
        context : UpdateContext
            Shared update context.

        Returns
        -------
        npt.NDArray[np.float64]
            Updated parameter ensemble array.

        Raises
        ------
        RuntimeError
            If strategy not initialized.
        """
        if self._smoother is None or self._cov_YY is None or self._D is None:
            raise RuntimeError("Strategy not initialized. Call initialize() first.")

        num_params = param_ensemble.shape[0]
        batch_size = _calculate_adaptive_batch_size(num_params, self._num_obs)
        batches = _split_by_batchsize(np.arange(0, num_params), batch_size)

        log_msg = (
            f"Running localization on {num_params} parameters, "
            f"{self._num_obs} responses, {context.ensemble_size} realizations "
            f"and {len(batches)} batches"
        )
        logger.info(log_msg)
        context.progress_callback(AnalysisStatusEvent(msg=log_msg))

        def progress_callback_wrapper(
            iterable: Sequence[T],
        ) -> TimedIterator[T]:
            return TimedIterator(iterable, context.progress_callback)

        start = time.time()

        for param_batch_idx in batches:
            update_idx = param_batch_idx[non_zero_variance_mask[param_batch_idx]]
            X_local = param_ensemble[update_idx, :]

            param_ensemble[update_idx, :] = self._smoother.assimilate(
                X=X_local,
                Y=context.responses,
                D=self._D,
                alpha=1.0,
                correlation_threshold=context.settings.correlation_threshold(
                    context.ensemble_size
                ),
                cov_YY=self._cov_YY,
                progress_callback=progress_callback_wrapper,
                n_jobs=NUM_JOBS_ADAPTIVE_LOC,
            )

        logger.info(
            f"Adaptive Localization of {param_group} completed "
            f"in {(time.time() - start) / 60} minutes"
        )

        return param_ensemble
