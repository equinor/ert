"""Standard ES update strategy without localization."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING

import humanize
import iterative_ensemble_smoother as ies
import numpy as np

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

    Pre-computes assimilation factors once during prepare() and
    applies updates to each parameter group via update().

    Parameters
    ----------
    enkf_truncation : float
        Singular value truncation threshold.
    rng : np.random.Generator
        Random number generator for reproducibility.
    progress_callback : Callable[[AnalysisEvent], None]
        Callback to report progress events.

    Raises
    ------
    ErtAnalysisError
        If preparing the assimilation fails.
    """

    def __init__(
        self,
        enkf_truncation: float,
        rng: np.random.Generator,
        progress_callback: Callable[[AnalysisEvent], None],
    ) -> None:
        self._enkf_truncation = enkf_truncation
        self._rng = rng
        self._progress_callback = progress_callback
        self._smoother: ies.ESMDA | None = None
        self._ensemble_size: int = 0
        self._num_obs: int = 0

    def prepare(self, obs_context: ObservationContext) -> None:
        """Pre-compute assimilation factors from observation data.

        Parameters
        ----------
        obs_context : ObservationContext
            Preprocessed observation and response data.
        """
        self._ensemble_size = obs_context.ensemble_size
        self._num_obs = obs_context.num_observations

        self._smoother = ies.ESMDA(
            covariance=obs_context.observation_errors**2,
            observations=obs_context.observation_values,
            alpha=1,
            seed=self._rng,
        )

        try:
            self._smoother.prepare_assimilation(
                Y=obs_context.responses,
                truncation=self._enkf_truncation,
                overwrite=True,
            )
        except Exception as err:
            raise ErtAnalysisError(
                "Failed while preparing assimilation, "
                "this might be due to outlier values in one "
                f"or more realizations: {err}"
            ) from err

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
        if self._smoother is None:
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

        X = param_ensemble[non_zero_variance_mask]
        param_ensemble[non_zero_variance_mask] = self._smoother.assimilate_batch(
            X=X, overwrite=True
        )

        elapsed = time.perf_counter() - start_time

        self._progress_callback(
            AnalysisStatusEvent(
                msg=f"Updated {param_config.name} ({param_config.type.upper()}) "
                f"in {humanize.precisedelta(timedelta(seconds=elapsed))}",
                detail=True,
            )
        )

        return param_ensemble
