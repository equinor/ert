"""Standard ES update strategy without localization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import iterative_ensemble_smoother as ies
import numpy as np
import scipy

from ert.analysis._update_commons import ErtAnalysisError
from ert.analysis.event import AnalysisErrorEvent, AnalysisStatusEvent, DataSection
from ert.analysis.snapshots import SmootherSnapshot

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.config import ParameterConfig

    from ._protocol import UpdateContext

logger = logging.getLogger(__name__)


class StandardESUpdate:
    """Standard ES update without localization.

    This strategy computes a transition matrix T such that:
        X_posterior = X_prior @ T

    The transition matrix is computed once during initialization and
    applied to all parameter groups.

    Parameters
    ----------
    smoother_snapshot : SmootherSnapshot
        Snapshot object for error reporting.

    Attributes
    ----------
    _T : npt.NDArray[np.float64] | None
        The computed transition matrix.
    """

    def __init__(self, smoother_snapshot: SmootherSnapshot) -> None:
        self._smoother_snapshot = smoother_snapshot
        self._T: npt.NDArray[np.float64] | None = None

    def initialize(self, context: UpdateContext) -> None:
        """Compute the transition matrix.

        This computes T = (I + K @ H) where K is the Kalman gain matrix.
        The identity is added to the diagonal for efficient computation.

        Parameters
        ----------
        context : UpdateContext
            Shared update context with observations and settings.

        Raises
        ------
        ErtAnalysisError
            If computing the transition matrix fails due to singular matrix.
        """
        smoother = ies.ESMDA(
            covariance=context.observation_errors**2,
            observations=context.observation_values,
            alpha=1,
            seed=context.rng,
            inversion=context.settings.inversion.lower(),
        )

        try:
            self._T = smoother.compute_transition_matrix(
                Y=context.responses,
                alpha=1.0,
                truncation=context.settings.enkf_truncation,
            )
        except scipy.linalg.LinAlgError as err:
            msg = (
                "Failed while computing transition matrix, "
                "this might be due to outlier values in one "
                f"or more realizations: {err}"
            )
            context.progress_callback(
                AnalysisErrorEvent(
                    error_msg=msg,
                    data=DataSection(
                        header=self._smoother_snapshot.header,
                        data=self._smoother_snapshot.csv,
                        extra=self._smoother_snapshot.extra,
                    ),
                )
            )
            raise ErtAnalysisError(msg) from err

        # Add identity in place for efficient computation: T = I + K @ H
        np.fill_diagonal(self._T, self._T.diagonal() + 1)

    def can_handle(self, param_config: ParameterConfig) -> bool:
        """Check if this strategy handles the parameter type.

        Parameters
        ----------
        param_config : ParameterConfig
            Configuration for the parameter.

        Returns
        -------
        bool
            Always True since standard update handles all types.
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
        """Apply the transition matrix to update parameters.

        Only parameters with non-zero variance are updated.

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
        if self._T is None:
            raise RuntimeError("Strategy not initialized. Call initialize() first.")

        num_obs = len(context.observation_values)
        log_msg = (
            f"There are {num_obs} responses and {context.ensemble_size} realizations."
        )
        logger.info(log_msg)
        context.progress_callback(AnalysisStatusEvent(msg=log_msg))

        # In-place multiplication is not yet supported, therefore avoiding @=
        param_ensemble[non_zero_variance_mask] @= self._T.astype(param_ensemble.dtype)

        return param_ensemble
