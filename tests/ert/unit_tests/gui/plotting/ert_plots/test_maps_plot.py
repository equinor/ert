import itertools

import pandas as pd
import pytest
from matplotlib.collections import TriMesh
from matplotlib.figure import Figure

from ert.gui.plotting.ert_plots import MapsPlot
from ert.gui.plotting.ert_plots.maps import (
    MEAN_SIMULATED_RESPONSE_MAP_TITLE,
    MISFIT_MAP_TITLE,
    OBSERVATIONS_MAP_TITLE,
)


def observation_data_with_locations() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            0: [1.0, 10.0, "0", 456000.00, 5938000.00],
            1: [1.0, 20.0, "5", 458000.00, 5940000.00],
            2: [1.0, 30.0, "8", 460000.00, 5936000.00],
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


@pytest.mark.parametrize(
    (
        "show_observations_map",
        "show_misfit_map",
        "show_mean_simulated_response_map",
        "expected_titles",
    ),
    [
        pytest.param(
            True,
            True,
            True,
            [
                OBSERVATIONS_MAP_TITLE,
                MISFIT_MAP_TITLE,
                MEAN_SIMULATED_RESPONSE_MAP_TITLE,
            ],
            id="all",
        ),
        pytest.param(
            True,
            False,
            True,
            [OBSERVATIONS_MAP_TITLE, MEAN_SIMULATED_RESPONSE_MAP_TITLE],
            id="no_misfits",
        ),
        pytest.param(
            False,
            True,
            True,
            [MISFIT_MAP_TITLE, MEAN_SIMULATED_RESPONSE_MAP_TITLE],
            id="no_observations",
        ),
        pytest.param(
            True,
            True,
            False,
            [OBSERVATIONS_MAP_TITLE, MISFIT_MAP_TITLE],
            id="no_mean_simulated_response",
        ),
    ],
)
def test_that_disabling_map_preserves_order_of_remaining_maps(
    make_plot_context,
    make_key_def,
    make_ensemble,
    show_observations_map,
    show_misfit_map,
    show_mean_simulated_response_map,
    expected_titles,
):
    ensemble = make_ensemble("ensemble")
    plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")
    plot_context.observations_map = show_observations_map
    plot_context.misfit_map = show_misfit_map
    plot_context.mean_simulated_response_map = show_mean_simulated_response_map

    figure = plot_maps(
        plot_context,
        make_key_def(key="SEISMIC_KEY", data_origin="seismic"),
        responses_for(ensemble),
        observation_data_with_locations(),
    )

    actual_titles = [axes.get_title() for axes in figure.axes if axes.get_title()]
    assert actual_titles == expected_titles

    axes_positions = [axes.get_position() for axes in figure.axes if axes.get_title()]
    assert all(
        left.x1 <= right.x0 for left, right in itertools.pairwise(axes_positions)
    )


def gridspec_columns(axes) -> int:
    return axes.get_subplotspec().get_topmost_subplotspec().get_gridspec().ncols


@pytest.mark.parametrize(
    ("map_selection_before", "map_selection_after"),
    [
        pytest.param(
            (True, True, True), (True, False, False), id="all_to_observations"
        ),
        pytest.param((True, True, True), (False, True, False), id="all_to_misfit"),
        pytest.param(
            (True, True, True),
            (False, False, True),
            id="all_to_mean_simulated_response",
        ),
        pytest.param((True, True, True), (True, True, False), id="all_to_two_maps"),
        pytest.param(
            (True, False, False), (True, True, True), id="observations_to_all"
        ),
    ],
)
def test_that_maps_widen_when_other_maps_are_deselected_and_narrow_when_selected(
    make_plot_context,
    make_key_def,
    make_ensemble,
    map_selection_before,
    map_selection_after,
):
    ensemble = make_ensemble("ensemble")

    def map_widths(map_selection):
        plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")
        (
            plot_context.observations_map,
            plot_context.misfit_map,
            plot_context.mean_simulated_response_map,
        ) = map_selection
        figure = plot_maps(
            plot_context,
            make_key_def(key="SEISMIC_KEY", data_origin="seismic"),
            responses_for(ensemble),
            observation_data_with_locations(),
        )
        return {
            axes.get_title(): axes.get_position(original=True).width
            for axes in figure.axes
            if axes.get_title()
        }

    widths_before = map_widths(map_selection_before)
    widths_after = map_widths(map_selection_after)

    maps_to_check = set(widths_before) & set(widths_after)
    assert maps_to_check != set()

    for title in maps_to_check:
        if len(widths_after) < len(widths_before):
            assert widths_after[title] > widths_before[title]
        else:
            assert widths_after[title] < widths_before[title]


def test_that_maps_plot_shows_no_map_selected_message_when_neither_map_is_selected(
    make_plot_context, make_key_def, make_ensemble
):
    ensemble = make_ensemble("ensemble")
    plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")
    plot_context.misfit_map = False
    plot_context.observations_map = False
    plot_context.mean_simulated_response_map = False

    figure = plot_maps(
        plot_context,
        make_key_def(key="SEISMIC_KEY", data_origin="seismic"),
        responses_for(ensemble),
        observation_data_with_locations(),
    )

    assert not figure.axes
    assert figure.texts[0].get_text().startswith("No map selected.")


def test_that_misfit_and_response_map_shows_no_data_message_when_ensemble_map_is_empty(
    make_plot_context, make_key_def
):
    figure = plot_maps(
        make_plot_context([], key="SEISMIC_KEY"),
        make_key_def(data_origin="seismic"),
        observation_data=observation_data_with_locations(),
    )

    misfit_map_axes = axes_with_title(figure, MISFIT_MAP_TITLE)
    mean_response_axes = axes_with_title(figure, MEAN_SIMULATED_RESPONSE_MAP_TITLE)
    assert misfit_map_axes.texts[0].get_text() == "No ensemble data available"
    assert mean_response_axes.texts[0].get_text() == "No ensemble data available"


@pytest.mark.parametrize(
    ("map_title", "expected_message"),
    [
        (MISFIT_MAP_TITLE, "No misfit data available"),
        (MEAN_SIMULATED_RESPONSE_MAP_TITLE, "No response data available"),
    ],
)
def test_that_ensemble_maps_show_no_data_message_for_unobserved_responses(
    make_plot_context, make_key_def, make_ensemble, map_title, expected_message
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

    map_axes = axes_with_title(figure, map_title)
    assert map_axes.texts[0].get_text() == expected_message


def test_that_map_tab_shows_no_obs_data_message_when_obs_data_is_empty(
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
    assert figure.texts[0].get_text() == "No observation data available"


def test_that_ensemble_maps_show_one_ensemble_limit_message_when_multiple_are_selected(
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
    mean_response_axes = axes_with_title(figure, MEAN_SIMULATED_RESPONSE_MAP_TITLE)
    assert (
        misfit_map_axes.texts[0].get_text()
        == "Multiple ensembles selected; this map supports one at a time"
    )
    assert (
        mean_response_axes.texts[0].get_text()
        == "Multiple ensembles selected; this map supports one at a time"
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


@pytest.mark.parametrize(
    ("show_observations_map", "show_misfit_map", "show_mean_simulated_response_map"),
    [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_that_only_leftmost_map_contains_yticks(
    make_plot_context,
    make_key_def,
    make_ensemble,
    show_observations_map,
    show_misfit_map,
    show_mean_simulated_response_map,
):
    ensemble = make_ensemble("ensemble")
    plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")
    plot_context.observations_map = show_observations_map
    plot_context.misfit_map = show_misfit_map
    plot_context.mean_simulated_response_map = show_mean_simulated_response_map

    figure = plot_maps(
        plot_context=plot_context,
        key_def=make_key_def(key="SEISMIC_KEY", data_origin="seismic"),
        ensemble_to_data_map=responses_for(ensemble),
        observation_data=observation_data_with_locations(),
    )

    map_axes = sorted(
        (axes for axes in figure.axes if axes.get_title()),
        key=lambda axes: axes.get_position().x0,
    )

    def y_ticklabels_visible(axes) -> bool:
        return all(tick.label1.get_visible() for tick in axes.yaxis.get_major_ticks())

    leftmost_map, *other_maps = map_axes
    assert y_ticklabels_visible(leftmost_map)
    assert leftmost_map.get_ylabel() == "north coordinate"
    for axes in other_maps:
        assert not axes.get_ylabel()
        assert not y_ticklabels_visible(axes), axes.get_title()


def test_that_ticks_are_using_coordinate_format(
    make_plot_context, make_key_def, make_ensemble
):
    ensemble = make_ensemble("ensemble")
    plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")
    figure = plot_maps(
        plot_context=plot_context,
        key_def=make_key_def(key="SEISMIC_KEY", data_origin="seismic"),
        ensemble_to_data_map=responses_for(ensemble),
        observation_data=observation_data_with_locations(),
    )
    yticks = [
        ytick.get_text()
        for ytick in axes_with_title(figure, OBSERVATIONS_MAP_TITLE).get_yticklabels()
        if ytick.get_text()
    ]
    x_ticks_observation_map = [
        xtick.get_text()
        for xtick in axes_with_title(figure, OBSERVATIONS_MAP_TITLE).get_xticklabels()
        if xtick.get_text()
    ]
    x_ticks_misfit_map = [
        xtick.get_text()
        for xtick in axes_with_title(figure, MISFIT_MAP_TITLE).get_xticklabels()
        if xtick.get_text()
    ]
    x_ticks_mean_simulated_response_map = [
        xtick.get_text()
        for xtick in axes_with_title(
            figure, MEAN_SIMULATED_RESPONSE_MAP_TITLE
        ).get_xticklabels()
        if xtick.get_text()
    ]

    assert yticks != []
    assert x_ticks_observation_map != []
    assert x_ticks_misfit_map != []
    assert x_ticks_mean_simulated_response_map != []

    assert all(label.endswith("M") for label in yticks)
    assert all(label.endswith("k") for label in x_ticks_observation_map)
    assert all(label.endswith("k") for label in x_ticks_misfit_map)
    assert all(label.endswith("k") for label in x_ticks_mean_simulated_response_map)


def test_that_mean_simulated_response_map_shows_the_realization_mean_per_location(
    make_plot_context, make_key_def, make_ensemble
):
    ensemble = make_ensemble("ensemble")
    plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")
    plot_context.observations_map = False
    plot_context.misfit_map = False

    ensemble_to_data_map = {
        ensemble: pd.DataFrame(
            data={"0": [1.0, 3.0], "5": [2.0, 6.0], "8": [10.0, 20.0]},
            index=pd.Index([0, 1], name="Realization"),
        )
    }

    figure = plot_maps(
        plot_context=plot_context,
        key_def=make_key_def(key="SEISMIC_KEY", data_origin="seismic"),
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data_with_locations(),
    )

    expected_mean_per_location = {
        (456000.0, 5938000.0): 2.0,
        (458000.0, 5940000.0): 4.0,
        (460000.0, 5936000.0): 15.0,
    }

    tripcolor = axes_with_title(figure, MEAN_SIMULATED_RESPONSE_MAP_TITLE).collections[
        0
    ]
    assert isinstance(tripcolor, TriMesh)

    eastings = tripcolor._triangulation.x
    northings = tripcolor._triangulation.y
    values = tripcolor.get_array()
    assert values is not None

    drawn_mean_per_location = {
        (easting, northing): value
        for easting, northing, value in zip(eastings, northings, values, strict=False)
    }

    assert drawn_mean_per_location == expected_mean_per_location
