"""Unit tests for compute_waterfall_data."""

from __future__ import annotations

import io
import logging
import uuid
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import polars.testing
import pytest

from ert.gui.plotting.waterfall_data import compute_waterfall_data
from ert.storage.blob_data import BlobStorageData, MatrixStorageData


# create dummy kalman gain blob
def _make_k_blob(
    parameter_group_sizes: dict[str, int] | None = None,
    name: str = "K",
) -> BlobStorageData:
    if parameter_group_sizes is None:
        parameter_group_sizes = {"PARAM": 1}
    return BlobStorageData(
        uri="k_matrix.npy",
        file_size=100,
        file_type="npy",
        name=name,
        blob_info=MatrixStorageData(
            update_algorithm="EnIF",
            sparse=False,
            shape=(1, 3),
            data_type="float64",
            parameter_group_sizes=parameter_group_sizes,
        ),
    )


def _dense_k_bytes(k_array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, k_array)
    return buf.getvalue()


@pytest.fixture
def k_blob_ensemble() -> MagicMock:
    """A fully wired mock ensemble that produces a valid waterfall result."""
    rng = np.random.default_rng(0)
    n_obs = 4
    n_real = 5

    k_row = rng.standard_normal(n_obs)
    k_array = k_row[np.newaxis, :]

    obs_vals = rng.standard_normal(n_obs) + 5.0
    simulated = rng.standard_normal((n_obs, n_real)) + 5.0
    real_cols = {str(i): simulated[:, i].tolist() for i in range(n_real)}

    prior = MagicMock()
    prior.get_realization_mask_with_responses.return_value = np.ones(n_real, dtype=bool)
    prior.get_observations_and_responses.return_value = pl.DataFrame(
        {
            "observation_key": [f"OBS_{i}" for i in range(n_obs)],
            "observations": obs_vals.tolist(),
            **real_cols,
        }
    )
    prior_vals = rng.standard_normal(n_real) + 1.0
    posterior_vals = prior_vals + rng.standard_normal(n_real) * 0.1
    prior.load_parameters_numpy.return_value = prior_vals[:, np.newaxis]

    ensemble = MagicMock()
    ensemble.id = uuid.uuid4()
    ensemble.parent = uuid.uuid4()
    ensemble.load_blobs.return_value = [_make_k_blob({"PARAM": 1})]
    ensemble.load_blob.return_value = _dense_k_bytes(k_array)
    ensemble.experiment.observation_keys = [f"OBS_{i}" for i in range(n_obs)]
    ensemble.load_parameters_numpy.return_value = posterior_vals[:, np.newaxis]
    ensemble._storage.get_ensemble.return_value = prior
    return ensemble


def test_that_missing_k_blob_returns_empty_dataframe():
    ensemble = MagicMock()
    ensemble.id = uuid.uuid4()
    ensemble.load_blobs.return_value = []

    assert compute_waterfall_data(ensemble, "PARAM").is_empty()


def test_that_empty_parameter_group_sizes_returns_empty_dataframe():
    ensemble = MagicMock()
    ensemble.id = uuid.uuid4()
    ensemble.load_blobs.return_value = [_make_k_blob(parameter_group_sizes={})]

    assert compute_waterfall_data(ensemble, "PARAM").is_empty()


def test_that_unknown_parameter_key_returns_empty_dataframe():
    ensemble = MagicMock()
    ensemble.id = uuid.uuid4()
    ensemble.load_blobs.return_value = [_make_k_blob({"OTHER": 1})]

    assert compute_waterfall_data(ensemble, "PARAM").is_empty()


def test_that_no_observation_keys_returns_empty_dataframe():
    prior = MagicMock()
    prior.get_realization_mask_with_responses.return_value = np.array([True, True])

    ensemble = MagicMock()
    ensemble.id = uuid.uuid4()
    ensemble.parent = uuid.uuid4()
    ensemble.load_blobs.return_value = [_make_k_blob({"PARAM": 1})]
    ensemble.load_blob.return_value = _dense_k_bytes(np.array([[1.0, 2.0]]))
    ensemble.experiment.observation_keys = []
    ensemble._storage.get_ensemble.return_value = prior

    assert compute_waterfall_data(ensemble, "PARAM").is_empty()


def test_that_k_row_width_and_observation_count_mismatch_returns_empty_dataframe(
    caplog,
):
    """K row has 3 columns but only 2 observations in the response DataFrame."""

    prior = MagicMock()
    prior.get_realization_mask_with_responses.return_value = np.array([True, True])
    prior.get_observations_and_responses.return_value = pl.DataFrame(
        {
            "observation_key": ["OBS_A", "OBS_B"],
            "observations": [1.0, 2.0],
            "0": [0.9, 1.9],
            "1": [1.1, 2.1],
        }
    )

    ensemble = MagicMock()
    ensemble.id = uuid.uuid4()
    ensemble.parent = uuid.uuid4()
    # K has 3 columns, but DataFrame only has 2 observations
    ensemble.load_blobs.return_value = [_make_k_blob({"PARAM": 1})]
    ensemble.load_blob.return_value = _dense_k_bytes(np.array([[1.0, 2.0, 3.0]]))
    ensemble.experiment.observation_keys = ["OBS_A", "OBS_B"]
    ensemble._storage.get_ensemble.return_value = prior

    with caplog.at_level(logging.WARNING, logger="ert.gui.plotting.waterfall_data"):
        result = compute_waterfall_data(ensemble, "PARAM")

    assert result.is_empty()
    assert (
        "Dimension mismatch: K row has 3 columns but only 2 observations" in caplog.text
    )


def test_that_result_has_prior_contributions_posterior_row_order(k_blob_ensemble):
    result = compute_waterfall_data(k_blob_ensemble, "PARAM")
    assert result["type"][0] == "prior"
    assert result["type"][-1] == "posterior"
    assert (result["type"][1:-1] == "contribution").all()


def test_that_prior_value_is_zero(k_blob_ensemble):
    result = compute_waterfall_data(k_blob_ensemble, "PARAM")
    assert result.filter(pl.col("type") == "prior")["value"][0] == pytest.approx(0.0)


def test_that_contributions_sum_to_posterior_value(k_blob_ensemble):
    result = compute_waterfall_data(k_blob_ensemble, "PARAM")
    prior_val = result.filter(pl.col("type") == "prior")["value"][0]
    posterior_val = result.filter(pl.col("type") == "posterior")["value"][0]
    contributions_sum = result.filter(pl.col("type") == "contribution")["value"].sum()
    assert prior_val + contributions_sum == pytest.approx(posterior_val, rel=1e-6)


def test_that_observations_exceeding_nobservations_limit_are_bundled(
    k_blob_ensemble,
):
    result = compute_waterfall_data(k_blob_ensemble, "PARAM", nobservations=2)
    contributions = result.filter(pl.col("type") == "contribution")
    assert "Other observations" in contributions["name"].to_list()
    # 2 named rows + 1 bundle row
    assert len(contributions) == 3


def test_that_all_observations_shown_when_within_nobservations_limit(
    k_blob_ensemble,
):
    result = compute_waterfall_data(k_blob_ensemble, "PARAM", nobservations=100)
    contributions = result.filter(pl.col("type") == "contribution")
    assert "Other observations" not in contributions["name"].to_list()


def test_that_flat_prior_uses_raw_mean_shift_as_posterior_value():
    """When prior std is below the threshold the raw mean shift is reported."""
    n_obs = 3
    k_row = np.array([0.5, -0.3, 0.2])
    obs_vals = np.array([10.0, 20.0, 30.0])
    # Simulated responses identical across realizations → S.mean == S
    simulated = np.tile([9.5, 19.5, 29.5], (5, 1)).T
    real_cols = {str(i): simulated[:, i].tolist() for i in range(5)}

    prior = MagicMock()
    prior.get_realization_mask_with_responses.return_value = np.ones(5, dtype=bool)
    prior.get_observations_and_responses.return_value = pl.DataFrame(
        {
            "observation_key": [f"OBS_{i}" for i in range(n_obs)],
            "observations": obs_vals.tolist(),
            **real_cols,
        }
    )
    flat_prior_val = 5.0
    prior.load_parameters_numpy.return_value = np.full((5, 1), flat_prior_val)

    ensemble = MagicMock()
    ensemble.id = uuid.uuid4()
    ensemble.parent = uuid.uuid4()
    ensemble.load_blobs.return_value = [_make_k_blob({"PARAM": 1})]
    ensemble.load_blob.return_value = _dense_k_bytes(k_row[np.newaxis, :])
    ensemble.experiment.observation_keys = [f"OBS_{i}" for i in range(n_obs)]
    ensemble.load_parameters_numpy.return_value = np.full((5, 1), flat_prior_val + 1.5)
    ensemble._storage.get_ensemble.return_value = prior

    result = compute_waterfall_data(ensemble, "PARAM")
    posterior_val = result.filter(pl.col("type") == "posterior")["value"][0]
    assert posterior_val == pytest.approx(1.5)


def test_that_k_row_offset_selects_the_correct_parameter_row():
    """With two parameters, the K row for the second parameter is used."""
    # Row 0 drives OBS_0, row 1 drives OBS_2 exclusively
    k_array = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    n_obs = 3
    n_real = 4

    obs_vals = np.array([1.0, 2.0, 3.0])
    # Simulated slightly below observed → positive innovation
    simulated = np.tile([1.5, 2.5, 3.5], (n_real, 1)).T
    real_cols = {str(i): simulated[:, i].tolist() for i in range(n_real)}

    prior = MagicMock()
    prior.get_realization_mask_with_responses.return_value = np.ones(n_real, dtype=bool)
    prior.get_observations_and_responses.return_value = pl.DataFrame(
        {
            "observation_key": [f"OBS_{i}" for i in range(n_obs)],
            "observations": obs_vals.tolist(),
            **real_cols,
        }
    )
    prior.load_parameters_numpy.return_value = np.ones((n_real, 1))

    ensemble = MagicMock()
    ensemble.id = uuid.uuid4()
    ensemble.parent = uuid.uuid4()
    ensemble.load_blobs.return_value = [_make_k_blob({"FIRST": 1, "SECOND": 1})]
    ensemble.load_blob.return_value = _dense_k_bytes(k_array)
    ensemble.experiment.observation_keys = [f"OBS_{i}" for i in range(n_obs)]
    ensemble.load_parameters_numpy.return_value = np.full((n_real, 1), 1.5)
    ensemble._storage.get_ensemble.return_value = prior

    result = compute_waterfall_data(ensemble, "SECOND")
    first_contribution_name = result.filter(pl.col("type") == "contribution")["name"][0]
    # SECOND's K row is [0, 0, 1] so only OBS_2 has a non-zero contribution
    assert first_contribution_name == "OBS_2"


def test_that_inactive_observations_are_excluded_by_status_filter():
    """Rows with status != 'Active' are filtered out before contribution computation."""
    # The K matrix is built from active observations only, so it has 1 column
    # matching the single active observation (OBS_0) that survives the filter.
    n_real = 4
    k_array = np.array([[1.0]])  # 1 column: active observation OBS_0

    prior = MagicMock()
    prior.get_realization_mask_with_responses.return_value = np.ones(n_real, dtype=bool)
    prior.get_observations_and_responses.return_value = pl.DataFrame(
        {
            "observation_key": ["OBS_0", "OBS_1"],
            "observations": [10.0, 20.0],
            "status": ["Active", "Inactive"],
            "0": [9.0, 19.0],
            "1": [9.0, 19.0],
            "2": [9.0, 19.0],
            "3": [9.0, 19.0],
        }
    )
    prior.load_parameters_numpy.return_value = np.ones((n_real, 1))

    ensemble = MagicMock()
    ensemble.id = uuid.uuid4()
    ensemble.parent = uuid.uuid4()
    ensemble.load_blobs.return_value = [_make_k_blob({"PARAM": 1})]
    ensemble.load_blob.return_value = _dense_k_bytes(k_array)
    ensemble.experiment.observation_keys = ["OBS_0", "OBS_1"]
    ensemble.load_parameters_numpy.return_value = np.full((n_real, 1), 1.5)
    ensemble._storage.get_ensemble.return_value = prior

    result = compute_waterfall_data(ensemble, "PARAM")
    observation_names = result.filter(pl.col("type") == "contribution")[
        "name"
    ].to_list()
    assert "OBS_1" not in observation_names
    assert "OBS_0" in observation_names
