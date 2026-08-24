import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ert.gui.plotting.ert_plots.cross import (
    _match_obs_to_responses,
    _observations_by_key_index,
    _plot_identity_line,
    _to_matchable_key,
)
from ert.gui.plotting.utils.plot_config import PlotConfig


def test_that_to_matchable_key_converts_numeric_values_to_float32():
    values = np.array([1, 2, 3])
    result = _to_matchable_key(values)
    assert result.dtype == np.float32
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_that_to_matchable_key_converts_non_numeric_values_to_string():
    values = np.array([1, "b", 3])
    result = _to_matchable_key(values)
    assert all(isinstance(value, str) for value in result)
    assert list(result) == ["1", "b", "3"]


def test_that_observations_by_key_index_drops_non_numeric_obs_values():
    obs_values = ["1", "2", "invalid", "4"]
    key_index = ["a", "b", "c", "d"]
    observation_data = pd.DataFrame([obs_values, key_index], index=["OBS", "key_index"])
    result = _observations_by_key_index(observation_data)
    assert len(result) == 3
    assert result.dtype == np.float32
    assert list(result) == [1.0, 2.0, 4.0]
    assert list(result.index) == ["a", "b", "d"]


def test_that_observations_by_key_index_drops_duplicate_key_indices():
    obs_values = ["1", "2", "3", "4"]
    key_index = ["a", "b", "b", "d"]
    observation_data = pd.DataFrame([obs_values, key_index], index=["OBS", "key_index"])
    result = _observations_by_key_index(observation_data)
    assert len(result) == 3
    assert result.index.is_unique
    assert result["b"] == pytest.approx(2.0)


def test_that_observations_by_key_index_maps_key_index_to_obs_values():
    obs_values = ["1", "2", "3"]
    key_index = ["a", "b", "c"]
    observation_data = pd.DataFrame([obs_values, key_index], index=["OBS", "key_index"])
    result = _observations_by_key_index(observation_data)
    assert len(result) == 3
    assert result["a"] == pytest.approx(1.0)
    assert result["b"] == pytest.approx(2.0)
    assert result["c"] == pytest.approx(3.0)


def test_that_match_obs_to_responses_returns_empty_arrays_when_no_common_keys():
    ensemble_data = pd.DataFrame([[1, 2], [3, 4]], columns=["x", "y"])
    obs_by_key_index = pd.Series([10, 20], index=["a", "b"])
    obs_flat, resp_flat = _match_obs_to_responses(ensemble_data, obs_by_key_index)
    assert len(obs_flat) == 0
    assert len(resp_flat) == 0


def test_that_match_obs_to_responses_returns_matched_obs_and_resp_values():
    ensemble_data = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    obs_by_key_index = pd.Series([10, 20], index=["a", "b"])
    obs_flat, resp_flat = _match_obs_to_responses(ensemble_data, obs_by_key_index)
    assert len(obs_flat) == 4
    assert len(resp_flat) == 4
    assert np.array_equal(obs_flat, np.array([10, 20, 10, 20]))
    assert np.array_equal(resp_flat, np.array([1, 2, 3, 4]))


def test_that_match_obs_to_responses_drops_pairs_where_response_is_nan():
    ensemble_data = pd.DataFrame([[1, np.nan], [np.nan, 4]], columns=["a", "b"])
    obs_by_key_index = pd.Series([10, 20], index=["a", "b"])
    obs_flat, resp_flat = _match_obs_to_responses(ensemble_data, obs_by_key_index)
    assert len(obs_flat) == 2
    assert len(resp_flat) == 2
    assert np.array_equal(obs_flat, np.array([10, 20]))
    assert np.array_equal(resp_flat, np.array([1, 4]))


def test_that_match_obs_to_responses_converts_response_values_to_numeric():
    ensemble_data = pd.DataFrame([["1", "2"], ["3", "4"]], columns=["a", "b"])
    obs_by_key_index = pd.Series([10, 20], index=["a", "b"])
    _, resp_flat = _match_obs_to_responses(ensemble_data, obs_by_key_index)
    assert np.array_equal(resp_flat, np.array([1.0, 2.0, 3.0, 4.0]))


def test_that_plot_identity_line_draws_identity_line_from_valid_input_data():

    config = PlotConfig()
    _, ax = plt.subplots()
    obs_values = np.array([1, 2, 3])
    resp_values = np.array([1, 2, 3])
    _plot_identity_line(ax, config, obs_values, resp_values)

    lines = ax.get_lines()
    assert len(lines) == 1
    line = lines[0]
    assert np.array_equal(line.get_xdata(), np.array([1, 3]))
    assert np.array_equal(line.get_ydata(), np.array([1, 3]))


@pytest.mark.parametrize(
    ("obs_values", "resp_values"),
    [
        (np.array([1]), np.array([1])),
        (np.array([-np.inf, 0, 1]), np.array([4, 5, 10])),
        (np.array([1, 2, np.inf]), np.array([1, 2, 3])),
    ],
)
def test_that_plot_identity_line_does_not_draw_identity_line_when_invalid_input(
    obs_values, resp_values
):
    config = PlotConfig()
    _, ax = plt.subplots()
    _plot_identity_line(ax, config, obs_values, resp_values)
    assert len(ax.get_lines()) == 0
