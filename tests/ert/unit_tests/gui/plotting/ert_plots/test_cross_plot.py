from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ert.gui.plotting.ert_plots.cross import (
    _match_obs_to_responses,
    _observations_by_key_index,
    _plot_cross,
    _plot_identity_line,
    _to_matchable_key,
    plotCross,
)
from ert.gui.plotting.plot_api import EnsembleObject
from ert.gui.plotting.utils.plot_config import PlotConfig
from ert.gui.plotting.utils.plot_context import PlotContext
from ert.gui.plotting.utils.plot_style import PlotStyle


@pytest.fixture
def make_ensemble():
    def _make(name: str, id_: str | None = None) -> EnsembleObject:
        return EnsembleObject(
            name,
            id_ if id_ is not None else name,
            False,
            "experiment",
            "2026-01-01T00:00:00",
        )

    return _make


@pytest.fixture
def make_plot_context():
    def _make(
        ensembles: list[EnsembleObject],
        *,
        key: str = "FOPR",
        plot_config: PlotConfig | None = None,
    ) -> PlotContext:
        return PlotContext(
            plot_config if plot_config is not None else PlotConfig(),
            ensembles=ensembles,
            ensembles_color_indexes=list(range(len(ensembles))),
            key=key,
            layer=None,
        )

    return _make


def test_that_plot_cross_shows_no_observations_message_when_observation_data_is_empty(
    make_ensemble, make_plot_context
):
    fig = plt.figure()
    ensemble = make_ensemble("ensemble1", "id1")
    plot_context = make_plot_context([ensemble])

    plotCross(
        fig,
        plot_context,
        ensemble_to_data_map=ensemble,
        observation_data=pd.DataFrame(),
    )

    assert fig.axes[0].texts[0].get_text() == "No observations available"
    assert not fig.axes[0].axison


def test_that_plot_cross_skips_ensembles_with_empty_data(
    make_ensemble, make_plot_context
):
    fig = plt.figure()
    ensemble1 = make_ensemble("ensemble1", "id1")
    plot_context = make_plot_context([ensemble1])
    ensemble_to_data_map = {
        ensemble1: pd.DataFrame(),
    }
    observation_data = pd.DataFrame(
        [["1", "2"], ["a", "b"]], index=["OBS", "key_index"]
    )

    plotCross(
        fig,
        plot_context,
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
    )

    assert len(fig.axes[0].get_lines()) == 0


def test_that_plot_cross_skips_ensembles_when_no_responses_match_observation_keys(
    make_ensemble, make_plot_context
):
    fig = plt.figure()
    ensemble1 = make_ensemble("ensemble1", "id1")
    plot_context = make_plot_context([ensemble1])
    ensemble_to_data_map = {
        ensemble1: pd.DataFrame([[1, 2], [3, 4]], columns=["x", "y"]),
    }
    observation_data = pd.DataFrame(
        [["10", "20"], ["a", "b"]], index=["OBS", "key_index"]
    )

    plotCross(
        fig,
        plot_context,
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
    )

    assert fig.axes[0].get_lines() == []


def test_that_plot_cross_draws_scatter_points_for_matching_observations_and_responses(
    make_ensemble, make_plot_context
):
    fig = plt.figure()
    ensemble1 = make_ensemble("ensemble1", "id1")
    plot_context = make_plot_context([ensemble1])
    ensemble_to_data_map = {
        ensemble1: pd.DataFrame([[1, 2]], columns=["a", "b"]),
    }
    observation_data = pd.DataFrame(
        [["10", "20"], ["a", "b"]], index=["OBS", "key_index"]
    )

    plotCross(
        fig,
        plot_context,
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
    )

    lines = fig.axes[0].get_lines()
    assert len(lines) == 2
    line = lines[0]
    assert np.array_equal(line.get_xdata(), np.array([10.0, 20.0]))
    assert np.array_equal(line.get_ydata(), np.array([1.0, 2.0]))


def test_that_plot_cross_draws_scatter_points_for_multiple_ensembles(
    make_ensemble, make_plot_context
):
    fig = plt.figure()
    ensemble1 = make_ensemble("ensemble1", "id1")
    ensemble2 = make_ensemble("ensemble2", "id2")
    plot_context = make_plot_context([ensemble1, ensemble2])
    ensemble_to_data_map = {
        ensemble1: pd.DataFrame([[1, 2]], columns=["a", "b"]),
        ensemble2: pd.DataFrame([[3, 4]], columns=["a", "b"]),
    }
    observation_data = pd.DataFrame(
        [["10", "20"], ["a", "b"]], index=["OBS", "key_index"]
    )

    plotCross(
        fig,
        plot_context,
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
    )

    lines = fig.axes[0].get_lines()
    assert len(lines) == 3
    line1 = lines[0]
    line2 = lines[1]
    assert np.array_equal(line1.get_xdata(), np.array([10.0, 20.0]))
    assert np.array_equal(line1.get_ydata(), np.array([1.0, 2.0]))
    assert np.array_equal(line2.get_xdata(), np.array([10.0, 20.0]))
    assert np.array_equal(line2.get_ydata(), np.array([3.0, 4.0]))


def test_that_plot_cross_switches_to_log_scale_when_configured(
    make_ensemble, make_plot_context
):
    fig = plt.figure()
    ensemble1 = make_ensemble("ensemble1", "id1")
    plot_context = make_plot_context([ensemble1])
    plot_context.log_scale = True
    ensemble_to_data_map = {
        ensemble1: pd.DataFrame([[1, 2]], columns=["a", "b"]),
    }
    observation_data = pd.DataFrame(
        [["10", "20"], ["a", "b"]], index=["OBS", "key_index"]
    )

    plotCross(
        fig,
        plot_context,
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
    )

    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


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


def test_that_plot_cross_uses_obs_values_as_x_and_resp_values_as_y():
    config = PlotConfig()
    _, ax = plt.subplots()
    obs_values = np.array([1, 2, 3])
    resp_values = np.array([4, 5, 6])
    label = "Test Label"
    _plot_cross(ax, config, obs_values, resp_values, label)

    lines = ax.get_lines()
    assert len(lines) == 1
    line = lines[0]
    assert np.array_equal(line.get_xdata(), obs_values)
    assert np.array_equal(line.get_ydata(), resp_values)


def test_that_plot_cross_draws_markers_without_a_connecting_line():
    config = PlotConfig()
    _, ax = plt.subplots()
    obs_values = np.array([1, 2, 3])
    resp_values = np.array([4, 5, 6])
    label = "Test Label"
    _plot_cross(ax, config, obs_values, resp_values, label)

    lines = ax.get_lines()
    assert len(lines) == 1
    line = lines[0]
    assert line.get_linestyle() == "None"


@pytest.mark.parametrize(
    ("style_marker", "expected_marker"),
    [(None, "o"), ("x", "x"), ("s", "s")],
)
def test_that_plot_cross_uses_style_marker_and_falls_back_to_circle_when_none(
    style_marker, expected_marker
):
    config = PlotConfig()
    _, ax = plt.subplots()
    obs_values = np.array([1, 2, 3])
    resp_values = np.array([4, 5, 6])
    label = "Test Label"
    style = PlotStyle("test")
    style.marker = style_marker

    with patch.object(config, "distribution_style", return_value=style):
        _plot_cross(ax, config, obs_values, resp_values, label)

    lines = ax.get_lines()
    assert len(lines) == 1
    line = lines[0]
    assert line.get_marker() == expected_marker


def test_that_plot_cross_forwards_label_to_the_drawn_line():
    config = PlotConfig()
    _, ax = plt.subplots()
    obs_values = np.array([1, 2, 3])
    resp_values = np.array([4, 5, 6])
    label = "Test Label"

    _plot_cross(ax, config, obs_values, resp_values, label)

    lines = ax.get_lines()
    assert len(lines) == 1
    line = lines[0]
    assert line.get_label() == label


def test_that_plot_cross_applies_color_and_alpha_from_distribution_style():
    config = PlotConfig()
    style = config.distribution_style()
    _, ax = plt.subplots()
    obs_values = np.array([1, 2, 3])
    resp_values = np.array([4, 5, 6])
    label = "Test Label"

    _plot_cross(ax, config, obs_values, resp_values, label)

    lines = ax.get_lines()
    assert len(lines) == 1
    line = lines[0]
    assert line.get_color() == style.color
    assert line.get_alpha() == style.alpha


def test_that_plot_cross_registers_the_drawn_line_as_a_legend_item_with_the_label():
    config = PlotConfig()
    with patch.object(config, "add_legend_item") as mock_add_legend_item:
        _, ax = plt.subplots()
        obs_values = np.array([1, 2, 3])
        resp_values = np.array([4, 5, 6])
        label = "Test Label"

        _plot_cross(ax, config, obs_values, resp_values, label)

        mock_add_legend_item.assert_called_once_with(label, ax.get_lines()[0])
