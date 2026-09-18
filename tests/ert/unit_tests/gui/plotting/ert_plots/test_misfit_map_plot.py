import pandas as pd
import pytest
from matplotlib.figure import Figure

from ert.gui.plotting.ert_plots import MisfitMapPlot
from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.plotting.utils import PlotConfig, PlotContext


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


@pytest.fixture
def make_key_def():
    def _make(
        *, key: str = "FOPR", data_origin: str = "seismic"
    ) -> PlotApiKeyDefinition:
        return PlotApiKeyDefinition(
            key,
            index_type=None,
            metadata={"data_origin": data_origin},
            observations=True,
            dimensionality=2,
        )

    return _make


def test_that_misfit_map_plot_show_no_data_message_when_ensemble_map_is_empty(
    make_plot_context,
    make_key_def,
) -> None:
    plot_context = make_plot_context([])
    ensemble_to_data_map = {}
    figure = Figure()

    MisfitMapPlot().plot(
        figure=figure,
        plot_context=plot_context,
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=pd.DataFrame(),
        key_def=make_key_def(),
        obs_loc=None,
        std_dev_images={},
    )

    assert len(figure.axes) == 1
    assert figure.axes[0].texts[0].get_text() == "No ensemble data available"


def test_that_misfit_map_plot_show_no_misfit_data_message_when_no_misfit_rows_produced(
    make_plot_context,
    make_key_def,
    make_ensemble,
) -> None:
    ensemble = make_ensemble("ensemble_1")
    plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")
    key_def = make_key_def(key="SEISMIC_KEY")
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
) -> None:
    ensemble = make_ensemble("ensemble_1")
    plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")
    key_def = make_key_def(key="SEISMIC_KEY")
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


def test_that_misfit_map_uses_default_title_and_axis_labels_when_plot_config_is_unset(
    make_plot_context,
    make_key_def,
    make_ensemble,
) -> None:
    ensemble = make_ensemble("ensemble_1")
    plot_context = make_plot_context(
        [ensemble],
        key="SEISMIC_KEY",
        plot_config=PlotConfig(title="Costum map title"),
    )
    key_def = make_key_def()
    figure = Figure()
    observation_data = pd.DataFrame(
        data={
            0: [1.0, 10.0, "0", 100.0, 200.0],
            1: [1.0, 20.0, "5", 150.0, 250.0],
            2: [1.0, 30.0, "8", 200.0, 100.0],
        },
        index=["STD", "OBS", "key_index", "EAST", "NORTH"],
    )

    ensemble_to_data_map = {
        ensemble: pd.DataFrame(
            data={"0": [1.0], "5": [2.0], "8": [3.0]},
            index=pd.Index([0], name="Realization"),
        )
    }

    MisfitMapPlot().plot(
        figure=figure,
        plot_context=plot_context,
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
        key_def=key_def,
        obs_loc=None,
        std_dev_images={},
    )

    assert len(figure.axes) == 2
    assert figure.axes[0].get_title() == "Costum map title"
    assert figure.axes[0].get_xlabel() == "east coordinate"
    assert figure.axes[0].get_ylabel() == "north coordinate"


def test_that_misfit_map_shows_message_when_more_than_one_ensemble_is_provided(
    make_plot_context,
    make_key_def,
    make_ensemble,
) -> None:
    ensemble_1 = make_ensemble("ensemble_1")
    ensemble_2 = make_ensemble("ensemble_2")
    plot_context = make_plot_context([ensemble_1, ensemble_2], key="SEISMIC_KEY")
    key_def = make_key_def(key="SEISMIC_KEY")
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


def test_that_misfit_map_pins_colorbar_range_when_plot_context_has_one(
    make_plot_context,
    make_key_def,
    make_ensemble,
) -> None:
    ensemble = make_ensemble("ensemble")
    plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")
    plot_context.colorbar_range = (-3.0, 7.0)

    observation_data = pd.DataFrame(
        data={
            0: [1.0, 10.0, "0", 100.00, 200.00],
            1: [1.0, 20.0, "5", 150.00, 250.00],
            2: [1.0, 30.0, "8", 200.00, 100.00],
        },
        index=["STD", "OBS", "key_index", "EAST", "NORTH"],
    )

    ensemble_to_data_map = {
        ensemble: pd.DataFrame(
            data={"0": [1.0], "5": [2.0], "8": [3.0]},
            index=pd.Index([0], name="Realization"),
        )
    }
    figure = Figure()

    MisfitMapPlot().plot(
        figure=figure,
        plot_context=plot_context,
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
        key_def=make_key_def(key="SEISMIC_KEY"),
        obs_loc=None,
        std_dev_images={},
    )

    tripcolor = figure.axes[0].collections[0]
    assert tripcolor.get_clim() == (-3.0, 7.0)
