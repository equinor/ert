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
import scipy.sparse as sp
from iterative_ensemble_smoother import AdaptiveESMDA

from ert.analysis.event import AnalysisEvent, AnalysisStatusEvent
from ert.storage import SparseMatrixArtifact

from ._batching import calculate_localization_batch_size, split_by_batch_size

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.config import ParameterConfig
    from ert.storage import Experiment

    from ._protocol import ObservationContext

logger = logging.getLogger(__name__)

# When running adaptive localization we reserve some resources for
# the GUI and other applications that might be running
RESERVED_CPU_CORES = 2
NUM_JOBS_ADAPTIVE_LOC = max(
    1, ((psutil.cpu_count(logical=False) or 1) - RESERVED_CPU_CORES)
)
ADAPTIVE_THRESHOLDED_CROSS_COVARIANCE_ARTIFACT = (
    "adaptive_localization/{parameter_group}/thresholded_cross_covariance"
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
        progress_callback: Callable[[AnalysisEvent], None],
        experiment: Experiment | None = None,
    ) -> None:
        self._correlation_threshold_fn = correlation_threshold
        self._enkf_truncation = enkf_truncation
        self._progress_callback = progress_callback
        self._experiment = experiment
        self._smoother: AdaptiveESMDA | None = None
        self._num_obs: int = 0
        self._ensemble_size: int = 0

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
        batch_size = calculate_localization_batch_size(num_params, self._num_obs)
        batches = split_by_batch_size(np.arange(0, num_params), batch_size)
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
        thresholded_blocks: list[sp.csr_array] = []
        thresholded_block_rows: list[npt.NDArray[np.int_]] = []

        def correlation_callback(
            corr_XY: npt.NDArray[np.float64],
            observations_per_parameter: npt.NDArray[np.int_],
        ) -> npt.NDArray[np.bool_]:
            mask = np.abs(corr_XY) > threshold
            thresholded_blocks.append(
                sp.csr_array(np.where(mask, corr_XY, 0.0), dtype=np.float32)
            )
            return mask

        start_time = time.perf_counter()
        for param_batch_idx in batches:
            update_idx = param_batch_idx[non_zero_variance_mask[param_batch_idx]]
            if update_idx.size == 0:
                continue
            X_local = param_ensemble[update_idx, :]
            thresholded_block_rows.append(update_idx)

            param_ensemble[update_idx, :] = self._smoother.assimilate_batch(
                X=X_local,
                correlation_callback=correlation_callback,
                overwrite=True,
                n_jobs=NUM_JOBS_ADAPTIVE_LOC,
            )
        elapsed = time.perf_counter() - start_time

        self._save_thresholded_cross_covariance(
            param_config.name,
            num_params,
            threshold,
            thresholded_blocks,
            thresholded_block_rows,
        )

        self._progress_callback(
            AnalysisStatusEvent(
                msg=f"Updated {param_config.name} ({param_config.type.upper()}) "
                f"in {humanize.precisedelta(timedelta(seconds=elapsed))}",
                detail=True,
            )
        )

        return param_ensemble

    def _save_thresholded_cross_covariance(
        self,
        parameter_group: str,
        num_params: int,
        threshold: float,
        blocks: list[sp.csr_array],
        block_rows: list[npt.NDArray[np.int_]],
    ) -> None:
        if self._experiment is None or not blocks:
            return

        matrix = sp.lil_array((num_params, self._num_obs), dtype=np.float32)
        updated_rows = np.concatenate(block_rows).astype(np.int64, copy=False)
        for rows, block in zip(block_rows, blocks, strict=True):
            matrix[rows, :] = block

        self._experiment.save_sparse_matrix(
            ADAPTIVE_THRESHOLDED_CROSS_COVARIANCE_ARTIFACT.format(
                parameter_group=parameter_group
            ),
            SparseMatrixArtifact(
                matrix=matrix.tocsr(),
                metadata={
                    "parameter_group": np.asarray(parameter_group),
                    "threshold": np.asarray(threshold, dtype=np.float64),
                    "num_parameters": np.asarray(num_params, dtype=np.int64),
                    "num_observations": np.asarray(self._num_obs, dtype=np.int64),
                    "updated_parameter_indices": updated_rows,
                },
            ),
        )
