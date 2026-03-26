"""Standard ES update strategy without localization."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import iterative_ensemble_smoother as ies
import numpy as np
import scipy

from ert.analysis._update_commons import ErtAnalysisError
from ert.analysis.event import (
    AnalysisEvent,
    AnalysisStatusEvent,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.config import ParameterConfig

    from ._protocol import ObservationContext


class StandardESUpdate:
    """Standard ES update without localization.

    This strategy computes a transition matrix T such that:
        X_posterior = X_prior @ T

    The transition matrix is computed once during prepare() and
    applied to all parameter groups via update().

    Parameters
    ----------
    inversion : str
        Inversion algorithm to use (e.g., "EXACT").
    enkf_truncation : float
        Singular value truncation threshold.
    rng : np.random.Generator
        Random number generator for reproducibility.
    progress_callback : Callable[[AnalysisEvent], None]
        Callback to report progress events.

    Attributes
    ----------
    _T : npt.NDArray[np.float64]
        The computed transition matrix (set after prepare()).

    Raises
    ------
    ErtAnalysisError
        If computing the transition matrix fails due to singular matrix.
    """

    def __init__(
        self,
        inversion: str,
        enkf_truncation: float,
        rng: np.random.Generator,
        progress_callback: Callable[[AnalysisEvent], None],
    ) -> None:
        self._inversion = inversion
        self._enkf_truncation = enkf_truncation
        self._rng = rng
        self._progress_callback = progress_callback
        self._T: npt.NDArray[np.float64] | None = None
        self._ensemble_size: int = 0
        self._num_obs: int = 0

    def prepare(self, obs_context: ObservationContext) -> None:
        """Compute the transition matrix from observation data.

        Parameters
        ----------
        obs_context : ObservationContext
            Preprocessed observation and response data.
        """
        self._ensemble_size = obs_context.ensemble_size
        self._num_obs = obs_context.num_observations

        smoother = ies.ESMDA(
            covariance=obs_context.observation_errors**2,
            observations=obs_context.observation_values,
            alpha=1,
            seed=self._rng,
            inversion=self._inversion.lower(),
        )

        try:
            self._T = smoother.compute_transition_matrix(
                Y=obs_context.responses,
                alpha=1.0,
                truncation=self._enkf_truncation,
            )
        except scipy.linalg.LinAlgError as err:
            raise ErtAnalysisError(
                "Failed while computing transition matrix, "
                "this might be due to outlier values in one "
                f"or more realizations: {err}"
            ) from err

        # Add identity in place for efficient computation: T = I + K @ H
        np.fill_diagonal(self._T, self._T.diagonal() + 1)

    def update(
        self,
        param_ensemble: npt.NDArray[np.floating],
        param_config: ParameterConfig,
        non_zero_variance_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.floating]:
        """Apply the transition matrix to update parameters.

        Only parameters with non-zero variance are updated.

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
        if self._T is None:
            raise RuntimeError("prepare() must be called before update()")

        num_params = param_ensemble.shape[0]
        self._progress_callback(
            AnalysisStatusEvent(
                msg=f"Updating {param_config.name} ({param_config.type.upper()}) "
                f"without localization, "
                f"{num_params} parameters, "
                f"{self._num_obs} observations, "
                f"{self._ensemble_size} realizations"
            )
        )

        start_time = time.perf_counter()
        param_ensemble[non_zero_variance_mask] @= self._T.astype(param_ensemble.dtype)
        elapsed = time.perf_counter() - start_time

        self._progress_callback(
            AnalysisStatusEvent(
                msg=f"Updated {param_config.name} ({param_config.type.upper()}) "
                f"in {elapsed:.2f}s",
                detail=True,
            )
        )

        return param_ensemble
