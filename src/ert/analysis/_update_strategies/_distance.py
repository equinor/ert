"""Distance-based localization update strategy."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from iterative_ensemble_smoother import LocalizedESMDA
from iterative_ensemble_smoother.utils import calc_rho_for_2d_grid_layer

from ert.analysis.event import AnalysisEvent, AnalysisStatusEvent
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


class DistanceLocalizationUpdate:
    """Distance-based localization for Field and Surface parameters."""

    def __init__(
        self,
        enkf_truncation: float,
        rng: np.random.Generator,
        param_type: type[Field | SurfaceConfig],
        progress_callback: Callable[[AnalysisEvent], None],
    ) -> None:
        self._enkf_truncation = enkf_truncation
        self._rng = rng
        self._param_type = param_type
        self._progress_callback = progress_callback
        self._obs_loc: ObservationLocations | None = None
        self._smoother: LocalizedESMDA | None = None
        self._ensemble_size: int = 0

    def prepare(self, obs_context: ObservationContext) -> None:
        if obs_context.observation_locations is None:
            raise RuntimeError(
                "Distance localization requires observation_locations in context"
            )
        self._obs_loc = obs_context.observation_locations
        self._ensemble_size = obs_context.ensemble_size
        self._smoother = LocalizedESMDA(
            covariance=self._obs_loc.observation_errors**2,
            observations=self._obs_loc.observation_values,
            alpha=1,
            seed=self._rng,
        )
        self._smoother.prepare_assimilation(
            Y=self._obs_loc.responses_with_loc,
            truncation=self._enkf_truncation,
            overwrite=True,
        )

    def update(
        self,
        param_ensemble: npt.NDArray[np.floating],
        param_config: ParameterConfig,
        non_zero_variance_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.floating]:
        if self._obs_loc is None or self._smoother is None:
            raise RuntimeError("prepare() must be called before update()")

        start = time.time()
        param_type_name = self._param_type.__name__
        log_msg = (
            f"Running distance localization on {param_type_name}"
            f" with {param_ensemble.shape[0]} parameters,"
            f" {self._obs_loc.xpos.shape[0]} observations,"
            f" {self._ensemble_size} realizations"
        )
        logger.info(log_msg)
        self._progress_callback(AnalysisStatusEvent(msg=log_msg))

        if self._param_type is Field:
            assert isinstance(param_config, Field)
            result = self._update_field(param_ensemble, param_config)
        else:
            assert isinstance(param_config, SurfaceConfig)
            result = self._update_surface(param_ensemble, param_config)

        logger.info(
            f"Distance Localization of {param_type_name} {param_config.name} completed "
            f"in {(time.time() - start) / 60} minutes"
        )
        return result

    def _update_field(
        self,
        param_ensemble: npt.NDArray[np.floating],
        param_config: Field,
    ) -> npt.NDArray[np.floating]:
        assert self._obs_loc is not None
        assert self._smoother is not None

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
            np.zeros_like(self._obs_loc.main_range),
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

        # Reshape 2D rho to (nx*ny, nobs), tile across nz layers for 3D fields
        rho_2d = rho_matrix.reshape(ertbox.nx * ertbox.ny, -1)
        rho_full = np.tile(rho_2d, (ertbox.nz, 1)) if ertbox.nz > 1 else rho_2d

        def localization_callback(
            K: npt.NDArray[np.floating],
        ) -> npt.NDArray[np.floating]:
            return K * rho_full

        return self._smoother.assimilate_batch(
            X=param_ensemble,
            localization_callback=localization_callback,
            overwrite=True,
        )

    def _update_surface(
        self,
        param_ensemble: npt.NDArray[np.floating],
        param_config: SurfaceConfig,
    ) -> npt.NDArray[np.floating]:
        assert self._obs_loc is not None
        assert self._smoother is not None

        xpos, ypos = transform_positions_to_local_field_coordinates(
            (param_config.xori, param_config.yori),
            param_config.rotation,
            self._obs_loc.xpos,
            self._obs_loc.ypos,
        )

        rotation_angle = transform_local_ellipse_angle_to_local_coords(
            param_config.rotation,
            np.zeros_like(self._obs_loc.main_range, dtype=np.floating),
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
            rotation_angle,
            right_handed_grid_indexing=False,
        )

        rho_flat = rho_matrix.reshape(-1, rho_matrix.shape[-1])

        def localization_callback(
            K: npt.NDArray[np.floating],
        ) -> npt.NDArray[np.floating]:
            return K * rho_flat

        return self._smoother.assimilate_batch(
            X=param_ensemble,
            localization_callback=localization_callback,
            overwrite=True,
        )
