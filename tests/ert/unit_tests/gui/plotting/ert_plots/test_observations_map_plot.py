import pandas as pd
from matplotlib.figure import Figure

from ert.gui.plotting.ert_plots.observations_map import ObservationsMapPlot


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
