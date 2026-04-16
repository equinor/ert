from contextlib import contextmanager
from datetime import datetime
from math import exp
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QToolButton
from pytestqt.qtbot import QtBot

from ert.config import ErtConfig, ObservationType
from ert.gui.main import _setup_main_window
from ert.gui.tools.event_viewer import GUILogHandler
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_window import ENSEMBLE, STD_DEV, PlotWindow
from ert.services import ErtServerController
from ert.storage import open_storage

from .conftest import get_child


def _breakthrough_config() -> ErtConfig:
    return ErtConfig.from_dict(
        {
            "NUM_REALIZATIONS": 8,
            "ENSPATH": "storage",
            "RUNPATH": "run_path/realization-<IENS>",
            "ECLBASE": "BASE",
            "OBS_CONFIG": (
                "obs_config",
                [
                    {
                        "type": ObservationType.BREAKTHROUGH,
                        "name": "BRT_OP1",
                        "KEY": "WWCT:OP1",
                        "ERROR": "3",
                        "DATE": "2000-01-06",
                        "THRESHOLD": 0.4,
                    }
                ],
            ),
        }
    )


def _iter_keys(key_model, data_model):
    for key_index in range(key_model.rowCount()):
        item = data_model.itemAt(data_model.index(key_index, 0))
        assert item is not None
        yield item, key_model.index(key_index, 0)


def select_plotter_figure(plot_window: PlotWindow, key: str, plot_tab_name: str):
    """the figure of the given key in the given plot tab"""
    central_tab = plot_window._central_tab
    data_types = get_child(plot_window, DataTypeKeysWidget)
    key_list = data_types.data_type_keys_widget

    found_selected_key = False
    for key_def, key_index in _iter_keys(key_list.model(), data_types.model):
        if key_def.key == key:
            key_list.setCurrentIndex(key_index)
            for tab_index, tab in enumerate(plot_window._plot_widgets):
                if tab.name == plot_tab_name:
                    found_selected_key = True
                    assert central_tab.isTabEnabled(tab_index)
                    central_tab.setCurrentWidget(tab)
                    assert key_def.dimensionality == tab._plotter.dimensionality
                    if plot_tab_name == STD_DEV:
                        # we need a better resolution for box plots
                        tab._figure.set_size_inches(
                            2000 / tab._figure.get_dpi(),
                            1000 / tab._figure.get_dpi(),
                        )
                    yield tab._figure.figure
    assert found_selected_key


def summary_response(realization: int) -> pl.DataFrame:
    num_points = 14
    center = 6.0 + 0.35 * realization
    steepness = 1.0 + 0.1 * (realization % 3)

    values = [
        0.01 + 0.94 / (1.0 + exp(-(day - center) / steepness))
        for day in range(1, num_points + 1)
    ]

    return pl.DataFrame(
        {
            "response_key": ["WWCT:OP1"] * num_points,
            "time": [
                datetime(2000, 1, day)  # noqa: DTZ001
                for day in range(1, num_points + 1)
            ],
            "values": pl.Series(values, dtype=pl.Float32),
        }
    )


@contextmanager
def open_plotter(config: ErtConfig, qtbot: QtBot):
    log_handler = GUILogHandler()
    with ErtServerController.init_service(project=Path(config.ens_path)):
        args_mock = Mock()
        args_mock.config = "breakthrough.ert"
        gui = _setup_main_window(config, args_mock, log_handler, config.ens_path)
        qtbot.addWidget(gui)

        button_plot_tool = get_child(gui, QToolButton, name="button_Create_plot")

        qtbot.mouseClick(button_plot_tool, Qt.MouseButton.LeftButton)
        plot_window = get_child(gui, PlotWindow)

        yield plot_window

        plot_window.close()


def setup_storage(config):
    def dump_all(configurations):
        return [c.model_dump(mode="json") for c in configurations]

    ens_config = config.ensemble_config
    num_reals = config.runpath_config.num_realizations
    with open_storage(config.ens_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": dump_all(ens_config.parameter_configuration),
                "response_configuration": dump_all(ens_config.response_configuration),
                "derived_response_configuration": dump_all(
                    ens_config.derived_response_configuration
                ),
                "observations": dump_all(config.observation_declarations),
                "ert_templates": config.ert_templates,
            },
            name="breakthrough-experiment",
        )
        ensemble = experiment.create_ensemble(ensemble_size=num_reals, name="prior")
        bt_config = experiment.derived_response_configuration["breakthrough"]
        for r in range(num_reals):
            ensemble.save_response("summary", summary_response(r), r)
            breakthrough_response = bt_config.derive_from_storage(0, r, ensemble)
            ensemble.save_response("breakthrough", breakthrough_response, r)


def create_breakthrough_figure(plot_tab_name: str):
    @pytest.fixture
    def plot_figure(use_tmpdir, qtbot: QtBot):
        config = _breakthrough_config()

        setup_storage(config)

        open_storage(config.ens_path, mode="r")

        with open_plotter(config, qtbot) as plot_window:
            yield from select_plotter_figure(
                plot_window, "BREAKTHROUGH:WWCT:OP1", plot_tab_name
            )

    return plot_figure


plot_figure = create_breakthrough_figure(ENSEMBLE)


@pytest.mark.mpl_image_compare(tolerance=10.0)
@pytest.mark.skip_mac_ci
@pytest.mark.snapshot_test
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_breakthrough_ensemble_visualization_matches_snapshot(plot_figure):
    return plot_figure
