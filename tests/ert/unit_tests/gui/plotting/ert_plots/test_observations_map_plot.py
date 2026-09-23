import pandas as pd
from matplotlib.figure import Figure

from ert.gui.plotting.ert_plots.observations_map import ObservationsMapPlot
from ert.gui.plotting.utils.plot_config import PlotConfig


def test_that_observations_map_shows_no_data_message_when_observation_data_is_empty(
    make_plot_context, make_key_def
):
    figure = Figure()
    plot_context = make_plot_context([])
    key_def = make_key_def(data_origin="summary")
    ObservationsMapPlot().plot(
        figure=figure,
        plot_context=plot_context,
        ensemble_to_data_map={},
        observation_data=pd.DataFrame(),
        std_dev_images={},
        obs_loc=None,
        key_def=key_def,
    )

    assert len(figure.axes) == 1
    assert figure.axes[0].texts[0].get_text() == "No observation data available"


def test_that_observations_map_renders_with_default_axis_labels_and_configured_title(
    make_plot_context, make_key_def
):
    figure = Figure()
    plot_context = make_plot_context(
        [], plot_config=PlotConfig(title="Observation map")
    )
    key_def = make_key_def(data_origin="summary")
    observation_data = pd.DataFrame(
        data={
            0: [100.0, 200.0, 10.0],
            1: [150.0, 250.0, 20.0],
            2: [200.0, 100.0, 30.0],
        },
        index=["EAST", "NORTH", "OBS"],
    )

    ObservationsMapPlot().plot(
        figure=figure,
        plot_context=plot_context,
        ensemble_to_data_map={},
        observation_data=observation_data,
        std_dev_images={},
        obs_loc=None,
        key_def=key_def,
    )

    assert len(figure.axes) == 2
    assert len(figure.axes[0].collections) > 0
    assert figure.axes[0].get_xlabel() == "east coordinate"
    assert figure.axes[0].get_ylabel() == "north coordinate"
    assert figure.axes[0].get_title() == "Observation map"
