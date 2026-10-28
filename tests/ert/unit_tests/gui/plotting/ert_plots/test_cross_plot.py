import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ert.gui.plotting.ert_plots.cross import (
    _plot_identity_line,
    plotCross,
)
from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.plotting.utils.plot_config import PlotConfig
from ert.gui.plotting.utils.plot_context import PlotContext


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


def make_plot_api():
    return PlotApiKeyDefinition(
        key="FOPR",
        index_type=None,
        observations=True,
        dimensionality=2,
        metadata={"data_origin": "seismic"},
    )


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
        key_def=make_plot_api(),
        response_type="seismic",
    )

    assert (
        fig.axes[0].texts[0].get_text()
        == "No observations or responses are available for selected data type key"
    )
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
        [[10.0, 20.0], [1.0, 1.0], ["1,1", "2,2"]],
        index=["OBS", "STD", "key_index"],
    )

    plotCross(
        fig,
        plot_context,
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
        key_def=make_plot_api(),
        response_type="seismic",
    )

    assert len(fig.axes[0].get_lines()) == 0


def test_that_plot_cross_skips_ensembles_when_no_responses_match_observation_keys(
    make_ensemble, make_plot_context
):
    fig = plt.figure()
    ensemble1 = make_ensemble("ensemble1", "id1")
    plot_context = make_plot_context([ensemble1])
    ensemble_to_data_map = {
        ensemble1: pd.DataFrame([[1, 2]], columns=["5,5", "3,3"]),
    }
    observation_data = pd.DataFrame(
        [[10, 20], [1.0, 2.0], ["1,1", "2,2"]], index=["OBS", "STD", "key_index"]
    )
    # }

    plotCross(
        fig,
        plot_context,
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
        key_def=make_plot_api(),
        response_type="seismic",
    )

    assert fig.axes[0].get_lines() == []


def test_that_plot_cross_draws_scatter_points_for_matching_observations_and_responses(
    make_ensemble, make_plot_context
):
    fig = plt.figure()
    ensemble1 = make_ensemble("ensemble1", "id1")
    plot_context = make_plot_context([ensemble1])
    ensemble_data = pd.DataFrame([[1, 2], [3, 4]], columns=["1,1", "2,2"])
    ensemble_data.index.name = "realization"
    ensemble_to_data_map = {
        ensemble1: ensemble_data,
    }
    observation_data = pd.DataFrame(
        [[10, 20], [1.0, 2.0], ["1,1", "2,2"]], index=["OBS", "STD", "key_index"]
    )

    plotCross(
        fig,
        plot_context,
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
        key_def=make_plot_api(),
        response_type="seismic",
    )

    lines = fig.axes[0].get_lines()
    assert len(lines) == 2
    line = lines[0]
    assert np.array_equal(line.get_xdata(), np.array([10.0, 10.0, 20.0, 20.0]))
    assert np.array_equal(line.get_ydata(), np.array([1.0, 3.0, 2.0, 4.0]))

    config = plot_context.plotConfig()
    style = config.distribution_style()
    assert line.get_color() == style.color
    assert line.get_alpha() == style.alpha
    assert "experiment : ensemble1" in config.legend_labels()
    assert line in config.legend_items()


def test_that_plot_cross_draws_scatter_points_for_multiple_ensembles(
    make_ensemble, make_plot_context
):
    fig = plt.figure()
    ensemble1 = make_ensemble("ensemble1", "id1")
    ensemble2 = make_ensemble("ensemble2", "id2")
    plot_context = make_plot_context([ensemble1, ensemble2])
    ensemble_to_data_map = {
        ensemble1: pd.DataFrame([[1, 2]], columns=["1,1", "2,2"]),
        ensemble2: pd.DataFrame([[30, 40]], columns=["1,1", "2,2"]),
    }
    observation_data = pd.DataFrame(
        [[10.0, 20.0], [1.0, 1.0], ["1,1", "2,2"]],
        index=["OBS", "STD", "key_index"],
    )

    plotCross(
        fig,
        plot_context,
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
        key_def=make_plot_api(),
        response_type="seismic",
    )

    lines = fig.axes[0].get_lines()
    assert len(lines) == 3
    line1 = lines[0]
    line2 = lines[1]
    assert np.array_equal(line1.get_xdata(), np.array([10.0, 20.0]))
    assert np.array_equal(line1.get_ydata(), np.array([1.0, 2.0]))
    assert np.array_equal(line2.get_xdata(), np.array([10.0, 20.0]))
    assert np.array_equal(line2.get_ydata(), np.array([30.0, 40.0]))

    identity_line = lines[-1]
    assert np.array_equal(identity_line.get_xdata(), np.array([1.0, 40.0]))
    assert np.array_equal(identity_line.get_ydata(), np.array([1.0, 40.0]))


def test_that_plot_cross_switches_to_log_scale_when_configured(
    make_ensemble, make_plot_context
):
    fig = plt.figure()
    ensemble1 = make_ensemble("ensemble1", "id1")
    plot_context = make_plot_context([ensemble1])
    plot_context.log_scale = True
    ensemble_to_data_map = {
        ensemble1: pd.DataFrame([[1, 2]], columns=["1,1", "2,2"]),
    }
    observation_data = pd.DataFrame(
        [[10.0, 20.0], [1.0, 1.0], ["1,1", "2,2"]],
        index=["OBS", "STD", "key_index"],
    )

    plotCross(
        fig,
        plot_context,
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
        key_def=make_plot_api(),
        response_type="seismic",
    )

    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


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
