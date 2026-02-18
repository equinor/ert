"""Distance-based localization update strategies."""

from __future__ import annotations

import abc
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from iterative_ensemble_smoother.experimental import DistanceESMDA
from iterative_ensemble_smoother.utils import calc_rho_for_2d_grid_layer

from ert.config import Field, SurfaceConfig
from ert.field_utils import (
    transform_local_ellipse_angle_to_local_coords,
    transform_positions_to_local_field_coordinates,
)

from ._protocol import ObservationLocations

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.config import ParameterConfig

    from ._protocol import ObservationContext

logger = logging.getLogger(__name__)


class _DistanceLocalizationBase(abc.ABC):
    """Base class for distance-based localization strategies.

    Provides shared initialization logic (smoother setup) and the
    update skeleton (log, compute rho, call smoother). Subclasses
    implement ``_compute_rho_and_update`` for their specific parameter type.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducibility.

    Attributes
    ----------
    _obs_loc : ObservationLocations | None
        Observation locations and ranges (set after prepare()).
    _smoother : DistanceESMDA | None
        The distance ESMDA smoother instance (set after prepare()).
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._obs_loc: ObservationLocations | None = None
        self._smoother: DistanceESMDA | None = None
        self._ensemble_size: int = 0

    def prepare(self, obs_context: ObservationContext) -> None:
        """Initialize smoother from observation context.

        Parameters
        ----------
        obs_context : ObservationContext
            Preprocessed observation and response data.

        Raises
        ------
        RuntimeError
            If obs_context.observation_locations is None.
        """
        if obs_context.observation_locations is None:
            raise RuntimeError(
                "Distance localization requires observation_locations in context"
            )

        self._obs_loc = obs_context.observation_locations
        self._ensemble_size = obs_context.ensemble_size
        self._smoother = DistanceESMDA(
            covariance=self._obs_loc.observation_errors**2,
            observations=self._obs_loc.observation_values,
            alpha=1,
            seed=self._rng,
        )

    @abc.abstractmethod
    def _compute_rho_and_update(
        self,
        param_ensemble: npt.NDArray[np.float64],
        param_config: ParameterConfig,
        smoother: DistanceESMDA,
    ) -> npt.NDArray[np.float64]:
        """Compute the correlation matrix and update parameters.

        Parameters
        ----------
        param_ensemble : npt.NDArray[np.float64]
            Parameter ensemble matrix.
        param_config : ParameterConfig
            Parameter configuration.
        smoother : DistanceESMDA
            Initialized smoother instance.

        Returns
        -------
        npt.NDArray[np.float64]
            Updated parameter ensemble.
        """

    def update(
        self,
        param_group: str,
        param_ensemble: npt.NDArray[np.float64],
        param_config: ParameterConfig,
        non_zero_variance_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float64]:
        """Update parameters using distance-based localization.

        Parameters
        ----------
        param_group : str
            Name of the parameter group.
        param_ensemble : npt.NDArray[np.float64]
            Parameter ensemble matrix (num_params x ensemble_size).
        param_config : ParameterConfig
            Parameter configuration.
        non_zero_variance_mask : npt.NDArray[np.bool_]
            Mask for parameters with non-zero variance.

        Returns
        -------
        npt.NDArray[np.float64]
            Updated parameter ensemble.

        Raises
        ------
        RuntimeError
            If prepare() was not called before update().
        """
        if self._obs_loc is None or self._smoother is None:
            raise RuntimeError("prepare() must be called before update()")

        param_type = type(param_config).__name__
        start = time.time()
        log_msg = (
            f"Running distance localization on {param_type}"
            f" with {param_ensemble.shape[0]} parameters,"
            f" {self._obs_loc.xpos.shape[0]} observations,"
            f" {self._ensemble_size} realizations"
        )
        logger.info(log_msg)

        param_ensemble = self._compute_rho_and_update(
            param_ensemble, param_config, self._smoother
        )

        logger.info(
            f"Distance Localization of {param_type} {param_group} completed "
            f"in {(time.time() - start) / 60} minutes"
        )

        return param_ensemble


class DistanceLocalizationFieldUpdate(_DistanceLocalizationBase):
    """Distance-based localization for Field parameters."""

    def _compute_rho_and_update(
        self,
        param_ensemble: npt.NDArray[np.float64],
        param_config: ParameterConfig,
        smoother: DistanceESMDA,
    ) -> npt.NDArray[np.float64]:
        if not isinstance(param_config, Field):
            raise TypeError(f"Expected Field config, got {type(param_config)}")

        ertbox = param_config.ertbox_params
        if ertbox.xinc is None:
            raise ValueError("Field grid resolution (xinc) must be defined")
        if ertbox.yinc is None:
            raise ValueError("Field grid resolution (yinc) must be defined")
        if ertbox.origin is None:
            raise ValueError("Field grid origin must be defined")
        if ertbox.rotation_angle is None:
            raise ValueError("Field grid rotation angle must be defined")

        xpos, ypos = transform_positions_to_local_field_coordinates(
            ertbox.origin,
            ertbox.rotation_angle,
            self._obs_loc.xpos,
            self._obs_loc.ypos,
        )

        ellipse_rotation = transform_local_ellipse_angle_to_local_coords(
            ertbox.rotation_angle,
            np.zeros_like(self._obs_loc.main_range, dtype=np.float64),
        )

        rho_matrix = calc_rho_for_2d_grid_layer(
            ertbox.nx,
            ertbox.ny,
            ertbox.xinc,
            ertbox.yinc,
            xpos,
            ypos,
            self._obs_loc.main_range,
            self._obs_loc.main_range,
            ellipse_rotation,
            right_handed_grid_indexing=True,
        )

        return smoother.update_params(
            X=param_ensemble,
            Y=self._obs_loc.responses_with_loc,
            rho_input=rho_matrix,
            nz=ertbox.nz,
        )


class DistanceLocalizationSurfaceUpdate(_DistanceLocalizationBase):
    """Distance-based localization for Surface parameters."""

    def _compute_rho_and_update(
        self,
        param_ensemble: npt.NDArray[np.float64],
        param_config: ParameterConfig,
        smoother: DistanceESMDA,
    ) -> npt.NDArray[np.float64]:
        if not isinstance(param_config, SurfaceConfig):
            raise TypeError(f"Expected SurfaceConfig, got {type(param_config)}")

        xpos, ypos = transform_positions_to_local_field_coordinates(
            (param_config.xori, param_config.yori),
            param_config.rotation,
            self._obs_loc.xpos,
            self._obs_loc.ypos,
        )

        rotation_angle_of_localization_ellipse = (
            transform_local_ellipse_angle_to_local_coords(
                param_config.rotation,
                np.zeros_like(self._obs_loc.main_range, dtype=np.float64),
            )
        )

        if param_config.yflip != 1:
            raise ValueError(
                f"Expected SurfaceConfig.yflip == 1, got {param_config.yflip}"
            )

        rho_matrix = calc_rho_for_2d_grid_layer(
            param_config.ncol,
            param_config.nrow,
            param_config.xinc,
            param_config.yinc,
            xpos,
            ypos,
            self._obs_loc.main_range,
            self._obs_loc.main_range,
            rotation_angle_of_localization_ellipse,
            right_handed_grid_indexing=False,
        )

        return smoother.update_params(
            X=param_ensemble,
            Y=self._obs_loc.responses_with_loc,
            rho_input=rho_matrix,
        )
