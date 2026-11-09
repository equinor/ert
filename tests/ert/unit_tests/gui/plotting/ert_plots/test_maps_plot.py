import pandas as pd
import pytest
from matplotlib.figure import Figure

from ert.gui.plotting.ert_plots import MapsPlot
from ert.gui.plotting.ert_plots.maps import MISFIT_MAP_TITLE, OBSERVATIONS_MAP_TITLE


def observation_data_with_locations() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            0: [1.0, 10.0, "0", 100.00, 200.00],
            1: [1.0, 20.0, "5", 150.00, 250.00],
            2: [1.0, 30.0, "8", 200.00, 100.00],
        },
        index=["STD", "OBS", "key_index", "EAST", "NORTH"],
    )


def responses_for(ensemble) -> dict:
    return {
        ensemble: pd.DataFrame(
            data={"0": [1.0], "5": [2.0], "8": [3.0]},
            index=pd.Index([0], name="Realization"),
        )
    }


def plot_maps(
    plot_context,
    key_def,
    ensemble_to_data_map=None,
    observation_data=None,
) -> Figure:
    figure = Figure()
    MapsPlot().plot(
        figure=figure,
        plot_context=plot_context,
        ensemble_to_data_map=ensemble_to_data_map or {},
        observation_data=(
            observation_data if observation_data is not None else pd.DataFrame()
        ),
        std_dev_images={},
        obs_loc=None,
        key_def=key_def,
    )
    return figure


def axes_with_title(figure: Figure, title: str):
    return next(axes for axes in figure.axes if axes.get_title() == title)


def test_that_maps_plot_places_the_observations_map_left_of_the_misfit_map(
    make_plot_context, make_key_def, make_ensemble
):
    ensemble = make_ensemble("ensemble")
    plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")

    figure = plot_maps(
        plot_context,
        make_key_def(key="SEISMIC_KEY", data_origin="seismic"),
        responses_for(ensemble),
        observation_data_with_locations(),
    )

    misfit_axes = axes_with_title(figure, MISFIT_MAP_TITLE)
    observations_axes = axes_with_title(figure, OBSERVATIONS_MAP_TITLE)
    assert observations_axes.get_position().x1 <= misfit_axes.get_position().x0


def gridspec_columns(axes) -> int:
    return axes.get_subplotspec().get_topmost_subplotspec().get_gridspec().ncols


@pytest.mark.parametrize(
    "selected_map_title", [MISFIT_MAP_TITLE, OBSERVATIONS_MAP_TITLE]
)
def test_that_the_selected_map_is_widened_when_the_other_map_is_deselected(
    make_plot_context, make_key_def, make_ensemble, selected_map_title
):
    ensemble = make_ensemble("ensemble")
    key_def = make_key_def(key="SEISMIC_KEY", data_origin="seismic")

    both_maps_figure = plot_maps(
        make_plot_context([ensemble], key="SEISMIC_KEY"),
        key_def,
        responses_for(ensemble),
        observation_data_with_locations(),
    )

    single_map_context = make_plot_context([ensemble], key="SEISMIC_KEY")
    if selected_map_title == MISFIT_MAP_TITLE:
        single_map_context.observations_map = False
    else:
        single_map_context.misfit_map = False
    single_map_figure = plot_maps(
        single_map_context,
        key_def,
        responses_for(ensemble),
        observation_data_with_locations(),
    )

    if selected_map_title == MISFIT_MAP_TITLE:
        assert (
            gridspec_columns(axes_with_title(both_maps_figure, MISFIT_MAP_TITLE)) == 2
        )
        assert (
            gridspec_columns(axes_with_title(single_map_figure, MISFIT_MAP_TITLE)) == 1
        )
    else:
        assert (
            gridspec_columns(axes_with_title(both_maps_figure, OBSERVATIONS_MAP_TITLE))
            == 2
        )
        assert (
            gridspec_columns(axes_with_title(single_map_figure, OBSERVATIONS_MAP_TITLE))
            == 1
        )


def test_that_maps_plot_shows_no_map_selected_message_when_neither_map_is_selected(
    make_plot_context, make_key_def, make_ensemble
):
    ensemble = make_ensemble("ensemble")
    plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")
    plot_context.misfit_map = False
    plot_context.observations_map = False

    figure = plot_maps(
        plot_context,
        make_key_def(key="SEISMIC_KEY", data_origin="seismic"),
        responses_for(ensemble),
        observation_data_with_locations(),
    )

    assert not figure.axes
    assert figure.texts[0].get_text().startswith("No map selected.")


def test_that_misfit_map_shows_no_data_message_when_ensemble_map_is_empty(
    make_plot_context, make_key_def
):
    figure = plot_maps(
        make_plot_context([], key="SEISMIC_KEY"),
        make_key_def(data_origin="seismic"),
        observation_data=observation_data_with_locations(),
    )

    misfit_map_axes = axes_with_title(figure, MISFIT_MAP_TITLE)
    assert misfit_map_axes.texts[0].get_text() == "No ensemble data available"


def test_that_misfit_map_shows_no_misfit_data_message_when_no_misfit_rows_produced(
    make_plot_context, make_key_def, make_ensemble
):
    ensemble = make_ensemble("ensemble_1")
    figure = plot_maps(
        plot_context=make_plot_context([ensemble], key="SEISMIC_KEY"),
        key_def=make_key_def(key="SEISMIC_KEY", data_origin="seismic"),
        ensemble_to_data_map={
            ensemble: pd.DataFrame(
                data={"99": [1.0], "100": [2.0]},
                index=pd.Index([0], name="Realization"),
            )
        },
        observation_data=observation_data_with_locations(),
    )

    misfit_map_axes = axes_with_title(figure, MISFIT_MAP_TITLE)
    assert misfit_map_axes.texts[0].get_text() == "No misfit data available"


def test_that_misfit_map_shows_no_obs_data_message_when_obs_data_is_empty(
    make_plot_context, make_key_def, make_ensemble
):
    ensemble = make_ensemble("ensemble_1")

    figure = plot_maps(
        plot_context=make_plot_context([ensemble], key="SEISMIC_KEY"),
        key_def=make_key_def(key="SEISMIC_KEY", data_origin="seismic"),
        ensemble_to_data_map={
            ensemble: pd.DataFrame(
                data={"99": [1.0]},
                index=pd.Index([0], name="Realization"),
            )
        },
    )

    misfit_map_axes = axes_with_title(figure, MISFIT_MAP_TITLE)
    assert misfit_map_axes.texts[0].get_text() == "No observation data available"


def test_that_misfit_map_shows_message_when_more_than_one_ensemble_is_provided(
    make_plot_context, make_key_def, make_ensemble
):
    ensemble_1 = make_ensemble("ensemble_1")
    ensemble_2 = make_ensemble("ensemble_2")

    figure = plot_maps(
        plot_context=make_plot_context([ensemble_1, ensemble_2], key="SEISMIC_KEY"),
        key_def=make_key_def(key="SEISMIC_KEY", data_origin="seismic"),
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
        observation_data=observation_data_with_locations(),
    )

    misfit_map_axes = axes_with_title(figure, MISFIT_MAP_TITLE)
    assert (
        misfit_map_axes.texts[0].get_text()
        == "Multiple ensembles selected; misfit map supports one at a time"
    )


def test_that_misfit_map_pins_colorbar_range_when_plot_context_has_one(
    make_plot_context, make_key_def, make_ensemble
):
    ensemble = make_ensemble("ensemble")
    plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")
    plot_context.colorbar_range = (-3.0, 7.0)

    figure = plot_maps(
        plot_context=plot_context,
        key_def=make_key_def(key="SEISMIC_KEY", data_origin="seismic"),
        ensemble_to_data_map=responses_for(ensemble),
        observation_data=observation_data_with_locations(),
    )

    tripcolor = axes_with_title(figure, MISFIT_MAP_TITLE).collections[0]
    assert tripcolor.get_clim() == (-3.0, 7.0)


def test_that_observations_map_shows_no_data_message_when_observation_data_is_empty(
    make_plot_context, make_key_def
):
    plot_context = make_plot_context([])
    plot_context.misfit_map = False

    figure = plot_maps(plot_context, make_key_def(data_origin="seismic"))

    observations_map_axes = axes_with_title(figure, OBSERVATIONS_MAP_TITLE)
    assert observations_map_axes.texts[0].get_text() == "No observation data available"
