import os
from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock

import pytest
from qtpy.QtCore import QTimer

from ert.config import ErtConfig
from ert.enkf_main import EnKFMain
from ert.gui.ertwidgets.customdialog import CustomDialog
from ert.gui.ertwidgets.listeditbox import ListEditBox
from ert.gui.ertwidgets.pathchooser import PathChooser
from ert.gui.main import GUILogHandler, _setup_main_window
from ert.services import StorageService
from ert.storage import open_storage

from .conftest import load_results_manually


@pytest.fixture
def well_file(tmp_path):
    (tmp_path / "OBS.txt").write_text("1.0 2.0 3.0 4.0")


@pytest.fixture
def ert_rft_setup(tmp_path):
    (tmp_path / "config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 3
            OBS_CONFIG obs
            GEN_DATA RFT_DATA INPUT_FORMAT:ASCII RESULT_FILE:rft_%d.txt REPORT_STEPS:0
            TIME_MAP time_map.txt
            """
        )
    )
    (tmp_path / "obs").write_text(
        dedent(
            """
            GENERAL_OBSERVATION RFT_OBS {
               DATA       = RFT_DATA;
               DATE       = 3001-09-01; -- The final odyssey
               VALUE      = 42.0;
               ERROR      = 0.69420;
            };
            """
        )
    )
    (tmp_path / "time_map.txt").write_text("3001-09-01\n")
    return tmp_path / "config.ert"


@pytest.fixture
def gen_data_in_runpath(tmp_path):
    simulations_dir = tmp_path / "simulations"
    simulations_dir.mkdir()
    for i in range(3):
        realization_dir = simulations_dir / f"realization-{i}"
        realization_dir.mkdir()
        (realization_dir / "iter-0").mkdir()
        (realization_dir / "iter-0" / "rft_0.txt").write_text(
            f"{i}.0", encoding="utf-8"
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_rft_csv_export_plugin_exports_rft_data(
    qtbot, ert_rft_setup, well_file, gen_data_in_runpath
):
    args = Mock()
    args.config = str(ert_rft_setup)

    output_file = Path("output.csv")

    ert_config = ErtConfig.from_file(args.config)
    enkf_main = EnKFMain(ert_config)
    with StorageService.init_service(
        ert_config=args.config,
        project=os.path.abspath(ert_config.ens_path),
    ), open_storage(ert_config.ens_path, mode="w") as storage:
        gui = _setup_main_window(enkf_main, args, GUILogHandler())
        qtbot.addWidget(gui)
        gui.notifier.set_storage(storage)
        gui.notifier.set_current_case(
            storage.create_experiment(
                parameters=enkf_main.ensembleConfig().parameter_configuration
            ).create_ensemble(
                name="default",
                ensemble_size=enkf_main.getEnsembleSize(),
            )
        )

        load_results_manually(qtbot, gui)

        def handle_rft_plugin_dialog():
            qtbot.waitUntil(lambda: gui.findChild(CustomDialog) is not None)
            dialog = gui.findChild(CustomDialog)
            assert isinstance(dialog, CustomDialog)
            trajectory_field = dialog.findChild(PathChooser, name="trajectory_chooser")
            assert isinstance(trajectory_field, PathChooser)
            trajectory_field._model.setValue(".")
            list_field = dialog.findChild(ListEditBox, name="list_of_cases")
            assert isinstance(list_field, ListEditBox)
            list_field._list_edit_line.setText("default")
            dialog.accept()

        QTimer.singleShot(1000, handle_rft_plugin_dialog)
        plugin_actions = gui.tools["Plugins"].getAction().menu().actions()
        rft_plugin = [
            a for a in plugin_actions if a.iconText() == "GEN_DATA RFT CSV Export"
        ][0]
        rft_plugin.trigger()
        qtbot.waitUntil(output_file.exists, timeout=20000)
        assert output_file.read_text(encoding="utf-8") == dedent(
            """\
            Realization,Well,Case,Iteration,Pressure
            0,OBS,default,0,0.0
            1,OBS,default,0,1.0
            2,OBS,default,0,2.0
            """
        )
