from pathlib import Path
from typing import cast

import pytest
import resfo
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QToolButton, QWidget
from pytestqt.qtbot import QtBot

from ert.config import ErtConfig
from ert.config.rft_config import RFTConfig
from ert.gui.tools.manage_experiments import ManageExperimentsPanel
from ert.gui.tools.manage_experiments.rft_qc_widget import RftQcWidget
from ert.gui.tools.manage_experiments.storage_info_widget import (
    _RealizationWidget,
    _RealizationWidgetTabs,
)
from ert.gui.tools.manage_experiments.storage_widget import StorageWidget
from ert.plugins import get_site_plugins
from ert.runpaths import Runpaths
from ert.storage import open_storage
from ert.storage.local_ensemble import _write_observation_metadata
from tests.ert import rft_qc_example
from tests.ert.ui_tests.gui.conftest import get_child, open_gui_with_config
from tests.ert.ui_tests.gui.test_docs_screenshots import GuiEvaluator
from tests.ert.unit_tests.gui.ertwidgets.test_rft_qc_widget import (
    display_all_points,
    pick_point,
)

REALIZATIONS = 10
ITERATIONS = 4

CONFIG = f"""\
QUEUE_SYSTEM LOCAL

NUM_REALIZATIONS {REALIZATIONS}

RUNPATH rft_demo_out/realization-<IENS>/iter-<ITER>
ENSPATH storage
ECLBASE {rft_qc_example.BASE_NAME}

OBS_CONFIG obs.txt
ZONEMAP {rft_qc_example.ZONEMAP_FILE}
APPROXIMATE_MISSING_RFT_VALUES TRUE

RFT WELL:WELL_A DATE:* PROPERTIES:*
RFT WELL:WELL_B DATE:2001-01-01 PROPERTIES:PRESSURE
"""


def _write_runpath_files(
    rft_config: RFTConfig, run_path: Path, realization: int, iteration: int
) -> None:
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / rft_qc_example.ZONEMAP_FILE).write_text(
        rft_qc_example.ZONEMAP, encoding="utf-8"
    )

    base = rft_config._rft_filepath(
        rft_config.input_files[0], str(run_path), realization, iteration
    )
    resfo.write(f"{base}.EGRID", rft_qc_example.egrid())
    resfo.write(f"{base}.RFT", rft_qc_example.rft_file())


def _create_storage(ert_config: ErtConfig) -> None:
    rft_config = cast(RFTConfig, ert_config.ensemble_config.response_configs["rft"])

    with open_storage(ert_config.ens_path, mode="w") as storage:
        experiment = storage.create_experiment(
            name="rft-qc-demo",
            experiment_config={
                "response_configuration": [
                    r.model_dump(mode="json")
                    for r in ert_config.ensemble_config.response_configuration
                ],
                "observations": [
                    o.model_dump(mode="json")
                    for o in ert_config.observation_declarations
                ],
                "shape_registry": ert_config.shape_registry.model_dump(mode="json"),
            },
        )
        ensembles = [
            storage.create_ensemble(
                experiment,
                ensemble_size=REALIZATIONS,
                iteration=iteration,
                name=f"iter-{iteration}",
            )
            for iteration in range(ITERATIONS)
        ]

        # Only populate the last iteration and response with files
        iteration = ITERATIONS - 1
        realization = REALIZATIONS - 1
        ensemble = ensembles[iteration]
        runpaths = Runpaths.from_config(ert_config)

        run_path = Path(runpaths.get_paths([realization], iteration)[0])
        _write_runpath_files(rft_config, run_path, realization, iteration)

        responses = rft_config.read_from_file(str(run_path), realization, iteration)
        _write_observation_metadata(str(run_path), realization, ensemble)
        ensemble.save_response(rft_config.type, responses, realization)


def _open_rft_qc_tab(qtbot: QtBot, gui: QWidget) -> RftQcWidget:
    qtbot.mouseClick(
        get_child(gui, QToolButton, name="button_Manage_experiments"),
        Qt.MouseButton.LeftButton,
    )
    panel = get_child(gui, ManageExperimentsPanel)

    storage_widget = get_child(panel, StorageWidget)
    storage_widget._tree_view.expandAll()
    model = storage_widget._tree_view.model()
    assert model is not None
    # experiment -> ensemble -> last realization
    storage_widget._tree_view.setCurrentIndex(
        model.index(0, 0, model.index(0, 0, model.index(0, 0)))
    )

    realization_widget = panel._storage_info_widget._content_layout.currentWidget()
    assert isinstance(realization_widget, _RealizationWidget)
    realization_widget._tab_widget.setCurrentIndex(
        _RealizationWidgetTabs.INSPECT_RFT_TAB
    )
    return realization_widget._rft_qc_widget


@pytest.mark.screenshot_test
@pytest.mark.filterwarnings("ignore:.*contains a RFT key but no forward model step")
def test_that_inspect_rft_tab_screenshot_is_up_to_date(
    qtbot,
    tmp_path: Path,
    monkeypatch,
    source_root,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rft.ert").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "obs.txt").write_text(
        rft_qc_example.observations_config(), encoding="utf-8"
    )

    ert_config = ErtConfig.with_plugins(get_site_plugins()).from_file(
        str(tmp_path / "rft.ert")
    )
    _create_storage(ert_config)

    with open_gui_with_config(tmp_path / "rft.ert") as gui:
        rft_qc_widget = _open_rft_qc_tab(qtbot, gui)
        rft_qc_widget._rft_file_label.setText("/path/to/file.RFT")
        gui.resize(2048, 768)
        display_all_points(rft_qc_widget)
        pick_point(rft_qc_widget._plot, 1)

        gui_evaluator = GuiEvaluator(
            source_root,
            "docs/ert/reference/configuration/fig",
            gui,
            qtbot,
        )
        gui_evaluator.compare_img_with_gui("gui_at_rft_qc_widget.png", 0.9999)
        assert not gui_evaluator.gui_change_detected(), gui_evaluator.change_report()
