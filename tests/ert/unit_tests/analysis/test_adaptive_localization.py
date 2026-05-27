from dataclasses import dataclass

import numpy as np
import pytest

from ert.analysis._update_strategies._adaptive import (
    ADAPTIVE_THRESHOLDED_CROSS_COVARIANCE_ARTIFACT,
    AdaptiveLocalizationUpdate,
)
from ert.analysis._update_strategies._protocol import ObservationContext
from ert.config import GenKwConfig
from ert.config.parameter_config import LocalizationType
from ert.storage import SparseMatrixArtifact


def _noop_callback(_event: object) -> None:
    pass


@dataclass
class _Experiment:
    sparse_matrices: dict[str, SparseMatrixArtifact]

    def save_sparse_matrix(
        self, name: str, sparse_matrix: SparseMatrixArtifact
    ) -> None:
        self.sparse_matrices[name] = sparse_matrix


def test_that_adaptive_localization_stores_thresholded_cross_covariance() -> None:
    responses = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    param_ensemble = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    obs_context = ObservationContext(
        responses=responses,
        observation_values=np.array([1.5, 1.5], dtype=np.float64),
        observation_errors=np.array([0.1, 0.1], dtype=np.float64),
        observation_perturbations=np.zeros_like(responses),
        observation_locations=None,
    )
    experiment = _Experiment({})
    updater = AdaptiveLocalizationUpdate(
        correlation_threshold=lambda _ensemble_size: 0.5,
        enkf_truncation=1.0,
        progress_callback=_noop_callback,
        experiment=experiment,
    )
    updater.prepare(obs_context)

    updater.update(
        param_ensemble.copy(),
        GenKwConfig(
            name="PARAM",
            group="PARAM",
            distribution={"name": "uniform", "min": 0, "max": 1},
            update_strategy=LocalizationType.ADAPTIVE,
        ),
        non_zero_variance_mask=np.array([True, False, True]),
    )

    artifact_name = ADAPTIVE_THRESHOLDED_CROSS_COVARIANCE_ARTIFACT.format(
        parameter_group="PARAM"
    )
    artifact = experiment.sparse_matrices[artifact_name]

    np.testing.assert_allclose(
        artifact.matrix.toarray(),
        np.array(
            [
                [1.0, -1.0],
                [0.0, 0.0],
                [-1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    assert artifact.matrix.dtype == np.float32
    assert artifact.metadata["parameter_group"] == "PARAM"
    assert artifact.metadata["threshold"] == pytest.approx(0.5)
    assert artifact.metadata["num_parameters"] == 3
    assert artifact.metadata["num_observations"] == 2
    np.testing.assert_array_equal(
        artifact.metadata["updated_parameter_indices"], np.array([0, 2])
    )
