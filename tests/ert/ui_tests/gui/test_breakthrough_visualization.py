from datetime import datetime
from math import exp
from unittest.mock import Mock

import polars as pl
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QToolButton
from pytestqt.qtbot import QtBot

from ert.config import ErtConfig, ObservationType
from ert.gui.main import _setup_main_window
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_window import ENSEMBLE, PlotWindow
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


def _mock_summary_response(realization: int) -> pl.DataFrame:
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
            "time": [datetime(2000, 1, day) for day in range(1, num_points + 1)],
            "values": pl.Series(values, dtype=pl.Float32),
        }
    )


@pytest.fixture
def plot_figure(use_tmpdir, qtbot: QtBot):
    config = _breakthrough_config()

    def dump_all(configurations):
        return [c.model_dump(mode="json") for c in configurations]

    with open_storage(config.ens_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": dump_all(
                    config.ensemble_config.parameter_configuration
                ),
                "response_configuration": dump_all(
                    config.ensemble_config.response_configuration
                ),
                "derived_response_configuration": dump_all(
                    config.ensemble_config.derived_response_configuration
                ),
                "observations": dump_all(config.observation_declarations),
                "ert_templates": config.ert_templates,
            },
            name="breakthrough-experiment",
        )
        ensemble = experiment.create_ensemble(
            ensemble_size=config.runpath_config.num_realizations,
            name="prior",
        )

        for realization in range(config.runpath_config.num_realizations):
            ensemble.save_response(
                "summary", _mock_summary_response(realization), realization
            )
            breakthrough_response = experiment.derived_response_configuration[
                "breakthrough"
            ].derive_from_storage(0, realization, ensemble)
            ensemble.save_response("breakthrough", breakthrough_response, realization)

    open_storage(config.ens_path, mode="r")

    with ErtServerController.init_service(
        project=config.ens_path,
    ):
        args_mock = Mock()
        args_mock.config = "breakthrough.ert"
        gui = _setup_main_window(config, args_mock, Mock(), config.ens_path)
        qtbot.addWidget(gui)

        button_plot_tool = get_child(gui, QToolButton, name="button_Create_plot")

        qtbot.mouseClick(button_plot_tool, Qt.MouseButton.LeftButton)
        plot_window = get_child(gui, PlotWindow)
        central_tab = plot_window._central_tab

        data_types = get_child(plot_window, DataTypeKeysWidget)
        key_list = data_types.data_type_keys_widget
        key_model = key_list.model()

        for key_index in range(key_model.rowCount()):
            to_select = data_types.model.itemAt(data_types.model.index(key_index, 0))
            assert to_select is not None
            if to_select.key == "BREAKTHROUGH:WWCT:OP1":
                key_list.setCurrentIndex(key_model.index(key_index, 0))
                selected_key = to_select
                for tab_index, tab in enumerate(plot_window._plot_widgets):
                    if tab.name == ENSEMBLE:
                        assert central_tab.isTabEnabled(tab_index)
                        central_tab.setCurrentWidget(tab)
                        assert (
                            selected_key.dimensionality == tab._plotter.dimensionality
                        )
                        yield tab._figure.figure

        plot_window.close()


@pytest.mark.mpl_image_compare(tolerance=10.0)
@pytest.mark.skip_mac_ci
@pytest.mark.snapshot_test
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_breakthrough_ensemble_visualization_matches_snapshot(plot_figure):
    return plot_figure
