"""Distance-based localization update strategy."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING

import humanize
import numpy as np
import scipy.sparse as sp
from iterative_ensemble_smoother import LocalizedESMDA

from ert.analysis.event import AnalysisEvent, AnalysisStatusEvent
from ert.config import Field, SurfaceConfig
from ert.field_utils import (
    AxisOrientation,
    calc_rho_for_2d_grid_layer,
    transform_local_ellipse_angle_to_local_coords,
    transform_positions_to_local_field_coordinates,
)
from ert.storage import SparseMatrixArtifact

from ._batching import calculate_localization_batch_size, split_by_batch_size
from ._protocol import ObservationLocations

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.config import ParameterConfig
    from ert.storage import Experiment

    from ._protocol import ObservationContext


class DistanceLocalizationUpdate:
    """Distance-based localization for Field and Surface parameters."""

    def __init__(
        self,
        enkf_truncation: float,
        param_type: type[Field | SurfaceConfig],
        progress_callback: Callable[[AnalysisEvent], None],
        experiment: Experiment | None = None,
    ) -> None:
        self._enkf_truncation = enkf_truncation
        self._param_type = param_type
        self._progress_callback = progress_callback
        self._experiment = experiment
        self._obs_loc: ObservationLocations | None = None
        self._smoother: LocalizedESMDA | None = None
        self._ensemble_size: int = 0
        self._num_obs: int = 0
        self._location_mask: npt.NDArray[np.bool_] | None = None

    def prepare(self, obs_context: ObservationContext) -> None:
        self._obs_loc = obs_context.observation_locations
        self._ensemble_size = obs_context.ensemble_size
        self._num_obs = obs_context.num_observations

        if self._obs_loc is None:
            self._location_mask = np.zeros(self._num_obs, dtype=bool)
        else:
            self._location_mask = self._obs_loc.location_mask

        self._smoother = LocalizedESMDA(
            covariance=obs_context.observation_errors**2,
            observations=obs_context.observation_values,
            alpha=1,
            # Observation perturbations are supplied explicitly below, so no
            # smoother RNG is used.
            seed=None,
        )
        self._smoother.prepare_assimilation(
            Y=obs_context.responses,
            truncation=self._enkf_truncation,
            overwrite=False,
            observation_perturbations=obs_context.observation_perturbations,
        )

    def update(
        self,
        param_ensemble: npt.NDArray[np.floating],
        param_config: ParameterConfig,
        non_zero_variance_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.floating]:
        if self._smoother is None or self._location_mask is None:
            raise RuntimeError("prepare() must be called before update()")

        num_params = param_ensemble.shape[0]
        num_located_obs = int(np.count_nonzero(self._location_mask))
        batch_size = calculate_localization_batch_size(num_params, self._num_obs)
        batches = split_by_batch_size(np.arange(0, num_params), batch_size)
        num_batches = len(batches)

        batch_info = f" and {num_batches} batches" if num_batches > 1 else ""
        self._progress_callback(
            AnalysisStatusEvent(
                msg=f"Updating {param_config.name} ({param_config.type.upper()}) "
                f"using distance-based localization, "
                f"{num_params} parameters, "
                f"{self._num_obs} observations "
                f"({num_located_obs} with locations), "
                f"{self._ensemble_size} realizations"
                f"{batch_info}"
            )
        )

        start_time = time.perf_counter()
        if self._param_type is Field:
            assert isinstance(param_config, Field)
            result = self._update_field(
                param_ensemble, param_config, batches, non_zero_variance_mask
            )
        else:
            assert isinstance(param_config, SurfaceConfig)
            result = self._update_surface(
                param_ensemble, param_config, batches, non_zero_variance_mask
            )
        elapsed = time.perf_counter() - start_time

        self._progress_callback(
            AnalysisStatusEvent(
                msg=f"Updated {param_config.name} ({param_config.type.upper()}) "
                f"in {humanize.precisedelta(timedelta(seconds=elapsed))}",
                detail=True,
            )
        )

        return result

    def _full_localization_matrix(
        self,
        num_params: int,
        located_rho: npt.NDArray[np.floating] | None,
    ) -> npt.NDArray[np.floating]:
        assert self._location_mask is not None

        rho = np.ones((num_params, self._num_obs), dtype=np.float32)
        if located_rho is not None:
            rho[:, self._location_mask] = located_rho.astype(np.float32, copy=False)
        return rho

    def _load_rho(
        self,
        parameter_group: str,
        nx: int,
        ny: int,
    ) -> sp.csr_array | None:
        if self._experiment is None or self._obs_loc is None:
            return None
        artifact = self._experiment.load_sparse_matrix(
            f"localization/{parameter_group}"
        )
        if artifact is None:
            return None
        rho = artifact.matrix
        grid_shape = tuple(artifact.metadata["grid_shape"])
        if grid_shape != (nx, ny):
            raise ValueError(
                f"Stored localization matrix for parameter group {parameter_group!r} "
                f"has grid shape {grid_shape}, expected {(nx, ny)}"
            )
        expected_num_rows = nx * ny
        if rho.shape[0] != expected_num_rows:
            raise ValueError(
                f"Stored localization matrix for parameter group {parameter_group!r} "
                f"has {rho.shape[0]} grid rows, "
                f"expected {expected_num_rows}"
            )

        stored_observation_key = artifact.metadata["observation_key"].astype(str)
        stored_observation_index = artifact.metadata["observation_index"].astype(str)
        if len(stored_observation_key) != rho.shape[1]:
            raise ValueError(
                f"Stored localization matrix for parameter group {parameter_group!r} "
                f"has {rho.shape[1]} observation columns, "
                f"but {len(stored_observation_key)} observation keys"
            )
        if len(stored_observation_index) != rho.shape[1]:
            raise ValueError(
                f"Stored localization matrix for parameter group {parameter_group!r} "
                f"has {rho.shape[1]} observation columns, "
                f"but {len(stored_observation_index)} observation indices"
            )

        stored_columns = {
            (key, index): column
            for column, (key, index) in enumerate(
                zip(stored_observation_key, stored_observation_index, strict=True)
            )
        }
        columns: list[int] = []
        for key, index in zip(
            self._obs_loc.observation_key.astype(str),
            self._obs_loc.observation_index.astype(str),
            strict=True,
        ):
            column = stored_columns.get((key, index))
            if column is None:
                return None
            columns.append(column)
        if columns == list(range(rho.shape[1])):
            return rho
        return rho[:, columns]

    def _save_rho(
        self,
        parameter_group: str,
        rho_matrix: sp.csr_array,
        nx: int,
        ny: int,
    ) -> None:
        if self._experiment is None or self._obs_loc is None:
            return
        self._experiment.save_sparse_matrix(
            f"localization/{parameter_group}",
            SparseMatrixArtifact(
                matrix=rho_matrix.astype(np.float32),
                metadata={
                    "grid_shape": np.asarray((nx, ny), dtype=np.int64),
                    "observation_key": self._obs_loc.observation_key,
                    "observation_index": self._obs_loc.observation_index,
                },
            ),
        )

    def _update_field(
        self,
        param_ensemble: npt.NDArray[np.floating],
        param_config: Field,
        batches: list[npt.NDArray[np.int_]],
        non_zero_variance_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.floating]:
        assert self._smoother is not None
        assert self._location_mask is not None

        # No located observations; use global assimilation
        if not self._location_mask.any():
            X = param_ensemble[non_zero_variance_mask]
            param_ensemble[non_zero_variance_mask] = self._smoother.assimilate_batch(
                X=X,
                overwrite=True,
            )
            return param_ensemble

        assert self._obs_loc is not None

        ertbox = param_config.ertbox_params
        if ertbox.xinc is None:
            raise ValueError("Field grid resolution (xinc) must be defined")
        if ertbox.yinc is None:
            raise ValueError("Field grid resolution (yinc) must be defined")
        if ertbox.origin is None:
            raise ValueError("Field grid origin must be defined")
        if ertbox.rotation_angle is None:
            raise ValueError("Field grid rotation angle must be defined")
        if ertbox.axis_orientation is None:
            raise ValueError("Field grid axis orientation must be defined")

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

        rho_matrix = self._load_rho(param_config.name, ertbox.nx, ertbox.ny)
        if rho_matrix is None:
            rho_matrix = sp.csr_array(
                calc_rho_for_2d_grid_layer(
                    nx=ertbox.nx,
                    ny=ertbox.ny,
                    xinc=ertbox.xinc,
                    yinc=ertbox.yinc,
                    obs_xpos=xpos,
                    obs_ypos=ypos,
                    obs_main_range=self._obs_loc.main_range,
                    obs_perp_range=self._obs_loc.main_range,
                    obs_anisotropy_angle=ellipse_rotation,
                    axis_orientation=ertbox.axis_orientation,
                ).reshape(ertbox.nx * ertbox.ny, -1)
            )
            self._save_rho(param_config.name, rho_matrix, ertbox.nx, ertbox.ny)

        for param_batch_idx in batches:
            update_idx = param_batch_idx[non_zero_variance_mask[param_batch_idx]]
            if update_idx.size == 0:
                continue
            xy_idx = update_idx // ertbox.nz
            rho_batch = self._full_localization_matrix(
                len(update_idx), rho_matrix[xy_idx, :].toarray()
            )

            def localization_callback(
                K: npt.NDArray[np.floating],
                rho: npt.NDArray[np.floating] = rho_batch,
            ) -> npt.NDArray[np.floating]:
                K *= rho
                return K

            param_ensemble[update_idx, :] = self._smoother.assimilate_batch(
                X=param_ensemble[update_idx, :],
                localization_callback=localization_callback,
                overwrite=True,
            )

        return param_ensemble

    def _update_surface(
        self,
        param_ensemble: npt.NDArray[np.floating],
        param_config: SurfaceConfig,
        batches: list[npt.NDArray[np.int_]],
        non_zero_variance_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.floating]:
        assert self._smoother is not None
        assert self._location_mask is not None

        # No located observations; use global assimilation
        if not self._location_mask.any():
            X = param_ensemble[non_zero_variance_mask]
            param_ensemble[non_zero_variance_mask] = self._smoother.assimilate_batch(
                X=X,
                overwrite=True,
            )
            return param_ensemble

        assert self._obs_loc is not None

        xpos, ypos = transform_positions_to_local_field_coordinates(
            (param_config.xori, param_config.yori),
            param_config.rotation,
            self._obs_loc.xpos,
            self._obs_loc.ypos,
        )

        rotation_angle = transform_local_ellipse_angle_to_local_coords(
            param_config.rotation,
            np.zeros_like(self._obs_loc.main_range, dtype=np.float64),
        )

        if param_config.yflip != 1:
            raise ValueError(
                f"Expected SurfaceConfig.yflip == 1, got {param_config.yflip}"
            )

        rho_matrix = self._load_rho(
            param_config.name, param_config.ncol, param_config.nrow
        )
        if rho_matrix is None:
            rho_matrix = sp.csr_array(
                calc_rho_for_2d_grid_layer(
                    nx=param_config.ncol,
                    ny=param_config.nrow,
                    xinc=param_config.xinc,
                    yinc=param_config.yinc,
                    obs_xpos=xpos,
                    obs_ypos=ypos,
                    obs_main_range=self._obs_loc.main_range,
                    obs_perp_range=self._obs_loc.main_range,
                    obs_anisotropy_angle=rotation_angle,
                    axis_orientation=AxisOrientation.LEFT_HANDED,
                ).reshape(param_config.ncol * param_config.nrow, -1)
            )
            self._save_rho(
                param_config.name, rho_matrix, param_config.ncol, param_config.nrow
            )
        for param_batch_idx in batches:
            update_idx = param_batch_idx[non_zero_variance_mask[param_batch_idx]]
            if update_idx.size == 0:
                continue
            rho_batch = self._full_localization_matrix(
                len(update_idx), rho_matrix[update_idx, :].toarray()
            )

            def localization_callback(
                K: npt.NDArray[np.floating],
                rho: npt.NDArray[np.floating] = rho_batch,
            ) -> npt.NDArray[np.floating]:
                K *= rho
                return K

            param_ensemble[update_idx, :] = self._smoother.assimilate_batch(
                X=param_ensemble[update_idx, :],
                localization_callback=localization_callback,
                overwrite=True,
            )

        return param_ensemble
