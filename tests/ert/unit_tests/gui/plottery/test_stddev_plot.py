from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from ert.gui.tools.plot.plot_api import EnsembleObject
from ert.gui.tools.plot.plot_types import ObservationPlotLocations
from ert.gui.tools.plot.plottery import PlotConfig, PlotContext
from ert.gui.tools.plot.plottery.plots import StdDevPlot


@pytest.fixture
def plot_context():
    context = Mock(spec=PlotContext)
    context.ensembles.return_value = [
        EnsembleObject(
            "ensemble_1", "id", False, "experiment_1", started_at="2012-12-10T00:00:00"
        )
    ]
    context.history_data = None
    context.layer = 0
    context.key.return_value = "FIELD"
    context.plotConfig.return_value = PlotConfig(title="StdDev Plot")
    return context


def test_stddev_plot_shows_boxplot(plot_context: PlotContext):
    rng = np.random.default_rng()
    figure = Figure()
    std_dev_data = rng.random((5, 5))
    obs_loc = ObservationPlotLocations(
        x=np.array([1, 3], dtype=np.float32),
        y=np.array([2, 4], dtype=np.float32),
        radius_x=np.ones(2, dtype=np.float64),
        radius_y=np.ones(2, dtype=np.float64),
        observation_key=np.full(2, "", dtype=str),
        observation_index=np.full(2, "", dtype=str),
    )
    StdDevPlot().plot(
        figure,
        plot_context,
        {},
        {},
        {"id": std_dev_data},
        obs_loc,
    )
    ax = figure.axes
    assert ax[0].get_title() == "experiment_1 : ensemble_1 layer=0"
    assert ax[1].get_ylabel() == "Standard deviation"
    annotation = [
        child for child in ax[1].get_children() if isinstance(child, plt.Annotation)
    ]
    assert len(annotation) == 1
    min_value = np.min(std_dev_data)
    mean_value = np.mean(std_dev_data)
    max_value = np.max(std_dev_data)
    assert (
        annotation[0].get_text()
        == f"Min: {min_value:.2f}\nMean: {mean_value:.2f}\nMax: {max_value:.2f}"
    )

    assert len(ax[0].images) == 1
    assert len(ax[0].collections) == 1


def test_stddev_plot_can_show_gaspari_cohn_localization_overlay_on_click(
    plot_context: PlotContext,
):
    figure = plt.figure()
    std_dev_data = np.ones((5, 5))
    obs_loc = ObservationPlotLocations(
        x=np.array([3], dtype=np.float64),
        y=np.array([3], dtype=np.float64),
        radius_x=np.array([1.0], dtype=np.float64),
        radius_y=np.array([1.0], dtype=np.float64),
        observation_key=np.array(["obs"], dtype=str),
        observation_index=np.array(["0"], dtype=str),
    )

    StdDevPlot().plot(
        figure,
        plot_context,
        {},
        {},
        {"id": std_dev_data},
        obs_loc,
    )

    ax = figure.axes
    assert len(ax[0].images) == 1
    assert len(ax[0].collections) == 1

    figure.canvas.draw()
    display_x, display_y = ax[0].transData.transform((2.5, 2.5))
    event = MouseEvent(
        "button_press_event",
        figure.canvas,
        display_x,
        display_y,
        MouseButton.LEFT,
    )
    figure.canvas.callbacks.process("button_press_event", event)

    overlay = ax[0].images[1].get_array()
    assert overlay.shape == (5, 5, 4)
    assert overlay.dtype == np.float32
    alpha = overlay[:, :, 3]
    max_alpha_index = np.unravel_index(np.argmax(alpha), alpha.shape)
    assert overlay[*max_alpha_index, :3].tolist() == pytest.approx(
        to_rgba("#ffb000")[:3]
    )
    assert 0.0 < alpha.max() <= 0.35
    assert alpha[0, 0] == pytest.approx(0.0)

    assert len(ax[0].images) == 2
    assert len(ax[0].collections) == 1
    assert ax[0].get_xlim() == pytest.approx((-0.5, 4.5))
    assert ax[0].get_ylim() == pytest.approx((4.5, -0.5))


def test_stddev_plot_maps_observation_overlay_to_rectangular_field_coordinates(
    plot_context: PlotContext,
):
    figure = plt.figure()
    std_dev_data = np.arange(15, dtype=np.float32).reshape(3, 5)
    obs_loc = ObservationPlotLocations(
        x=np.array([2], dtype=np.float64),
        y=np.array([4], dtype=np.float64),
        radius_x=np.array([1.0], dtype=np.float64),
        radius_y=np.array([2.0], dtype=np.float64),
        observation_key=np.array(["obs"], dtype=str),
        observation_index=np.array(["0"], dtype=str),
    )

    StdDevPlot().plot(
        figure,
        plot_context,
        {},
        {},
        {"id": std_dev_data},
        obs_loc,
    )

    ax = figure.axes[0]
    assert ax.images[0].get_array().shape == (5, 3)

    figure.canvas.draw()
    display_x, display_y = ax.transData.transform((1.5, 3.5))
    event = MouseEvent(
        "button_press_event",
        figure.canvas,
        display_x,
        display_y,
        MouseButton.LEFT,
    )
    figure.canvas.callbacks.process("button_press_event", event)

    assert ax.images[1].get_array().shape == (5, 3, 4)
    assert len(ax.images) == 2
    assert len(ax.collections) == 1
    assert ax.get_xlim() == pytest.approx((-0.5, 2.5))
    assert ax.get_ylim() == pytest.approx((4.5, -0.5))


def test_stddev_plot_prefers_persisted_rho_overlay(plot_context: PlotContext):
    figure = plt.figure()
    std_dev_data = np.ones((3, 3), dtype=np.float32)
    obs_loc = ObservationPlotLocations(
        x=np.array([2], dtype=np.float64),
        y=np.array([2], dtype=np.float64),
        radius_x=np.array([1.0], dtype=np.float64),
        radius_y=np.array([1.0], dtype=np.float64),
        observation_key=np.array(["obs"], dtype=str),
        observation_index=np.array(["0"], dtype=str),
    )
    rho = np.zeros((3, 3), dtype=np.float32)
    rho[0, 1] = 1.0
    localization_provider = Mock(return_value=rho)

    StdDevPlot().plot(
        figure,
        plot_context,
        {},
        {},
        {"id": std_dev_data},
        obs_loc,
        localization_provider,
    )

    ax = figure.axes[0]
    figure.canvas.draw()
    display_x, display_y = ax.transData.transform((1.5, 1.5))
    event = MouseEvent(
        "button_press_event",
        figure.canvas,
        display_x,
        display_y,
        MouseButton.LEFT,
    )
    figure.canvas.callbacks.process("button_press_event", event)

    localization_provider.assert_called_once_with("FIELD", "id", "obs", "0")
    overlay = ax.images[1]
    assert overlay.get_array().shape == (3, 3)
    assert overlay.get_array()[1, 0] == pytest.approx(1.0)
    assert overlay.get_array().sum() == pytest.approx(1.0)
    assert overlay.cmap(0.0) == pytest.approx((*to_rgba("#ffb000")[:3], 0.0))
    assert overlay.cmap(1.0) == pytest.approx((*to_rgba("#ffb000")[:3], 0.35))
    assert len(ax.images) == 2
    assert len(ax.collections) == 1


def test_that_stddev_plot_does_not_crash_and_returns_early_when_no_ensembles():
    figure = Figure()
    context = Mock(spec=PlotContext)
    context.ensembles.return_value = []
    context.layer = 0
    StdDevPlot().plot(
        figure,
        context,
        {},
        {},
        {},
        None,
    )
    assert len(figure.axes) == 0
