from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import resfo
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QToolButton
from pytestqt.qtbot import QtBot

from ert.config import ErtConfig
from ert.config.parsing.observations_parser import ObservationType
from ert.gui.main import GUILogHandler, _setup_main_window
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_ensemble_selection_widget import (
    EnsembleSelectListWidget,
)
from ert.gui.tools.plot.plot_window import (
    ENSEMBLE,
    STATISTICS,
    PlotWindow,
)
from ert.services import ErtServerConnection
from ert.storage import open_storage
from tests.ert.rft_generator import cell_start

from .conftest import (
    add_experiment_manually,
    get_child,
    load_results_manually,
    wait_for_child,
)


def pad_to(lst: list[int], target_len: int):
    return np.pad(lst, (0, target_len - len(lst)), mode="constant")


def write_egrid(path: Path):
    resfo.write(
        path,
        [
            ("FILEHEAD", pad_to([3, 2007, 0, 0, 0, 0, 1], 100)),
            ("MAPAXES ", np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=">f4")),
            ("GRIDUNIT", np.array([b"METRES  ", b"        "], dtype="|S8")),
            ("GRIDHEAD", pad_to([1, 1, 1, 1], 100)),
            (
                "COORD   ",
                np.array(
                    [
                        [[[0, 0, 0], [0, 0, 10]], [[0, 1, 0], [0, 1, 10]]],
                        [[[1, 0, 0], [1, 0, 10]], [[1, 1, 0], [1, 1, 10]]],
                    ],
                    dtype=">f4",
                ).ravel(),
            ),
            ("ZCORN   ", np.array([0, 0, 0, 0, 10, 10, 10, 10], dtype=">f4")),
            ("ACTNUM  ", np.ones((8,), dtype=">i4")),
            ("ENDGRID ", np.array([], dtype=">i4")),
        ],
    )


@pytest.fixture
def rft_config(tmp_path: Path):
    num_realizations = 2
    for i in range(num_realizations):
        runpath = tmp_path / "run_path" / f"realization-{i}"
        runpath.mkdir(parents=True)
        rft_file = runpath / "BASE.RFT"
        offset = i / 2
        depth = np.linspace(0, 2 * np.pi, dtype=np.float32)
        write_egrid(runpath / "BASE.EGRID")
        resfo.write(
            rft_file,
            [
                *cell_start(
                    date=(1, 1, 2000),
                    well_name="WELL",
                    ijks=[(i, i, i) for i in range(50)],
                ),
                ("PRESSURE", np.sin(depth)),
                ("DEPTH   ", depth + offset),
            ],
        )
    return ErtConfig.from_dict(
        {
            "NUM_REALIZATIONS": num_realizations,
            "ENSPATH": str(tmp_path / "storage"),
            "RUNPATH": str(tmp_path / "run_path/realization-<IENS>"),
            "ECLBASE": "BASE",
            "OBS_CONFIG": (
                "obs_config",
                [
                    {
                        "type": ObservationType.RFT,
                        "name": f"name{i}",
                        "WELL": "WELL",
                        "VALUE": 1 / (i + 1),
                        "ERROR": "0.1",
                        "DATE": "2000-01-01",
                        "PROPERTY": "PRESSURE",
                        "NORTH": 0.5,
                        "EAST": 0.5,
                        "TVD": 0.5 * i,
                    }
                    for i in range(1, 13)
                ],
            ),
        }
    )


@pytest.fixture(
    params=[
        ("WELL:2000-01-01:PRESSURE", ENSEMBLE),
        ("WELL:2000-01-01:PRESSURE", STATISTICS),
    ],
)
def plot_figure(qtbot: QtBot, request, rft_config: ErtConfig):
    key, plot_name = request.param
    args_mock = Mock()
    open_storage(rft_config.ens_path, mode="r")
    log_handler = GUILogHandler()
    with (
        ErtServerConnection.init_service(
            project=rft_config.ens_path,
        ),
    ):
        gui = _setup_main_window(
            rft_config, args_mock, log_handler, rft_config.ens_path
        )
        qtbot.addWidget(gui)
        add_experiment_manually(qtbot, gui)
        load_results_manually(qtbot, gui)

        button_plot_tool = gui.findChild(QToolButton, "button_Create_plot")
        assert button_plot_tool
        qtbot.mouseClick(button_plot_tool, Qt.MouseButton.LeftButton)
        plot_window = wait_for_child(gui, qtbot, PlotWindow)
        central_tab = plot_window._central_tab

        data_types = plot_window.findChild(DataTypeKeysWidget)
        key_list = data_types.data_type_keys_widget
        key_model = key_list.model()
        assert key_model is not None

        case_selection = get_child(
            plot_window, EnsembleSelectListWidget, "ensemble_selector"
        )
        # select all ensembles
        for index in range(case_selection.count()):
            assert (item := case_selection.item(index))
            if not item.data(Qt.ItemDataRole.CheckStateRole):
                case_selection.slot_toggle_plot(item)

        found_selected_key = False
        for i in range(key_model.rowCount()):
            to_select = data_types.model.itemAt(data_types.model.index(i, 0))
            assert to_select is not None
            if to_select.key == key:
                index = key_model.index(i, 0)
                key_list.setCurrentIndex(index)
                selected_key = to_select
                for i, tab in enumerate(plot_window._plot_widgets):
                    if tab.name == plot_name:
                        found_selected_key = True
                        if central_tab.isTabEnabled(i):
                            central_tab.setCurrentWidget(tab)
                            assert (
                                selected_key.dimensionality
                                == tab._plotter.dimensionality
                            )
                            yield tab._figure.figure
                        else:
                            assert (
                                selected_key.dimensionality
                                != tab._plotter.dimensionality
                            )
        assert found_selected_key
        plot_window.close()


# We had an issue where the mpl_image_compare decorator
# was put on an inner function. That makes any failure not
# report so it has to be on a top level test.
@pytest.mark.mpl_image_compare(tolerance=10.0)
@pytest.mark.skip_mac_ci  # test is slow
def test_that_all_rft_visualizations_matches_snapshot(plot_figure):
    return plot_figure
