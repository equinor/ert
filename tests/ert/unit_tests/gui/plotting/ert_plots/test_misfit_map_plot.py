import pandas as pd
from matplotlib.figure import Figure

from ert.gui.plotting.ert_plots import MisfitMapPlot


def test_that_misfit_map_plot_show_no_data_message_when_ensemble_map_is_empty(
    make_plot_context,
    make_key_def,
):
    plot_context = make_plot_context([])
    ensemble_to_data_map = {}
    figure = Figure()

    MisfitMapPlot().plot(
        figure=figure,
        plot_context=plot_context,
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=pd.DataFrame(),
        key_def=make_key_def(data_origin="seismic"),
        obs_loc=None,
        std_dev_images={},
    )

    assert len(figure.axes) == 1
    assert figure.axes[0].texts[0].get_text() == "No ensemble data available"


def test_that_misfit_map_plot_show_no_misfit_data_message_when_no_misfit_rows_produced(
    make_plot_context,
    make_key_def,
    make_ensemble,
):
    ensemble = make_ensemble("ensemble_1")
    plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")
    key_def = make_key_def(key="SEISMIC_KEY", data_origin="seismic")
    figure = Figure()
    observation_data = pd.DataFrame(
        data={0: [1.0, 10.0, "0"], 1: [10.0, 200.0, "5"]},
        index=["STD", "OBS", "key_index"],
    )

    MisfitMapPlot().plot(
        figure=figure,
        plot_context=plot_context,
        ensemble_to_data_map={
            ensemble: pd.DataFrame(
                data={"99": [1.0], "100": [2.0]},
                index=pd.Index([0], name="Realization"),
            )
        },
        observation_data=observation_data,
        key_def=key_def,
        obs_loc=None,
        std_dev_images={},
    )

    assert len(figure.axes) == 1
    assert figure.axes[0].texts[0].get_text() == "No misfit data available"


def test_that_misfit_map_plot_show_no_obs_data_message_when_obs_data_is_empty(
    make_plot_context,
    make_key_def,
    make_ensemble,
):
    ensemble = make_ensemble("ensemble_1")
    plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")
    key_def = make_key_def(key="SEISMIC_KEY", data_origin="seismic")
    figure = Figure()
    observation_data = pd.DataFrame()  # Empty observation data

    MisfitMapPlot().plot(
        figure=figure,
        plot_context=plot_context,
        ensemble_to_data_map={
            ensemble: pd.DataFrame(
                data={"99": [1.0]},
                index=pd.Index([0], name="Realization"),
            )
        },
        observation_data=observation_data,
        key_def=key_def,
        obs_loc=None,
        std_dev_images={},
    )

    assert len(figure.axes) == 1
    assert figure.axes[0].texts[0].get_text() == "No observation data available"


def test_that_misfit_map_shows_message_when_more_than_one_ensemble_is_provided(
    make_plot_context,
    make_key_def,
    make_ensemble,
):
    ensemble_1 = make_ensemble("ensemble_1")
    ensemble_2 = make_ensemble("ensemble_2")
    plot_context = make_plot_context([ensemble_1, ensemble_2], key="SEISMIC_KEY")
    key_def = make_key_def(key="SEISMIC_KEY", data_origin="seismic")
    figure = Figure()
    observation_data = pd.DataFrame(
        data={0: [1.0, 10.0, "0"], 1: [10.0, 200.0, "5"]},
        index=["STD", "OBS", "key_index"],
    )

    MisfitMapPlot().plot(
        figure=figure,
        plot_context=plot_context,
        ensemble_to_data_map={
            ensemble_1: pd.DataFrame(
                data={"99": [1.0]},
                index=pd.Index([0], name="Realization"),
            ),
            ensemble_2: pd.DataFrame(
                data={"100": [2.0]},
                index=pd.Index([0], name="Realization"),
            ),
        },
        observation_data=observation_data,
        key_def=key_def,
        obs_loc=None,
        std_dev_images={},
    )

    assert len(figure.axes) == 1
    assert (
        figure.axes[0].texts[0].get_text()
        == "Multiple ensembles selected; misfit map supports one at a time"
    )
