from pathlib import Path
from typing import cast

import pytest
import resfo
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QWidget

from ert.config import ErtConfig
from ert.config.rft_config import RFTConfig
from ert.gui.experiments import ExperimentPanel, RunDialog
from ert.gui.experiments.manual_update_panel import ManualUpdatePanel
from ert.gui.experiments.view.update import ReportLogTable, UpdateWidget
from ert.plugins import get_site_plugins
from ert.run_models.manual_update import ManualUpdate
from ert.sample_prior import sample_prior
from tests.ert.rft_generator import create_egrid, rft_entry
from tests.ert.ui_tests.gui.conftest import (
    ENSEMBLE_NAME,
    add_experiment_manually,
    get_child,
    load_results_manually,
    open_gui_with_config,
    wait_for_child,
)
from tests.ert.unit_tests.config.summary_generator import simple_smspec, simple_unsmry

NUM_REALIZATIONS = 2
ECLBASE = "BASE"
GRID = create_egrid(1, 1, 1, 100, 100, 100)
CONFIG = f"""\
QUEUE_SYSTEM LOCAL
NUM_REALIZATIONS {NUM_REALIZATIONS}
RUNPATH realization-<IENS>/iter-<ITER>
ECLBASE {ECLBASE}
OBS_CONFIG obs.txt
GEN_KW COEFFS coeff_priors
SUMMARY FOPR
RFT WELL:WELL_B DATE:2000-01-01 PROPERTIES:PRESSURE,SWAT
"""
OBSERVATIONS = """\
SUMMARY_OBSERVATION FOPR_OBS {
    VALUE = 5.629901e16;
    ERROR = 1e15;
    DATE  = 2014-01-01;
    KEY   = FOPR;
};

RFT_OBSERVATION RFT_OBS {
    WELL = WELL_B;
    DATE = 2000-01-01;
    PROPERTY = PRESSURE;
    VALUE = 120;
    ERROR = 10;
    EAST = 50;
    NORTH = 50;
    TVD = 50;
};
"""


def _write_runpath_files(
    rft_config: RFTConfig, runpath: Path, realization: int
) -> None:
    runpath.mkdir(parents=True, exist_ok=True)

    base = rft_config._rft_filepath(
        rft_config.input_files[0], str(runpath), realization, 0
    )
    resfo.write(f"{base}.EGRID", GRID)
    resfo.write(
        f"{base}.RFT",
        # No "swat" value given, which is requested by the RFT
        # config above, so this triggers an ObservationReportWarning.
        rft_entry(
            well_name=b"WELL_B",
            date=(1, 1, 2000),
            ijks=[(1, 1, 1)],
            pressure=[120.0 + realization],
            depth=[50.0],
        ),
    )

    smspec = simple_smspec()
    unsmry = simple_unsmry()
    # Perturb slightly per realization so the ensemble has non-zero spread
    unsmry.steps[0].ministeps[0].params[1] += realization * 1e13
    smspec.to_file(str(runpath / f"{ECLBASE}.SMSPEC"))
    unsmry.to_file(str(runpath / f"{ECLBASE}.UNSMRY"))


@pytest.mark.usefixtures("use_tmpdir")
def test_that_report_pane_shows_warning_only_for_observations_with_warnings(qtbot):
    Path("config.ert").write_text(CONFIG, encoding="utf-8")
    Path("obs.txt").write_text(OBSERVATIONS, encoding="utf-8")
    Path("coeff_priors").write_text("A UNIFORM 0 1\n", encoding="utf-8")

    ert_config = ErtConfig.with_plugins(get_site_plugins()).from_file("config.ert")
    rft_config = cast(RFTConfig, ert_config.ensemble_config.response_configs["rft"])

    with open_gui_with_config("config.ert") as gui:
        add_experiment_manually(qtbot, gui)

        with gui.notifier.write_storage() as storage:
            experiment = storage.get_experiment_by_name("My_experiment")
            ensemble = experiment.get_ensemble_by_name(ENSEMBLE_NAME)
            sample_prior(
                ensemble,
                range(NUM_REALIZATIONS),
                random_seed=1,
                num_realizations=NUM_REALIZATIONS,
            )

        for realization in range(NUM_REALIZATIONS):
            runpath = Path(f"realization-{realization}") / "iter-0"
            _write_runpath_files(rft_config, runpath, realization)

        load_results_manually(qtbot, gui)

        experiment_panel = get_child(gui, ExperimentPanel)
        simulation_mode_combo = get_child(experiment_panel, QComboBox)
        simulation_mode_combo.setCurrentText(ManualUpdate.name())

        manual_update_panel = get_child(experiment_panel, ManualUpdatePanel)
        manual_update_panel._experiment_name_field.setText("update-with-warning")

        run_experiment_button = get_child(
            experiment_panel, QWidget, name="run_experiment"
        )
        qtbot.mouseClick(run_experiment_button, Qt.MouseButton.LeftButton)

        run_dialog = wait_for_child(gui, qtbot, RunDialog)
        qtbot.waitUntil(lambda: run_dialog.is_experiment_done() is True, timeout=20000)

        update_widget = get_child(run_dialog, UpdateWidget)
        report_table = get_child(update_widget, ReportLogTable)

        headers = report_table.data.header
        status_col = headers.index("status")
        obs_key_col = headers.index("observation_key")

        rows_by_obs_key = {
            row[obs_key_col]: i for i, row in enumerate(report_table.data.data)
        }

        fopr_item = report_table.item(rows_by_obs_key["FOPR_OBS"], status_col)
        rft_item = report_table.item(rows_by_obs_key["RFT_OBS"], status_col)
        assert fopr_item is not None
        assert rft_item is not None

        assert fopr_item.text() == "Active"
        assert not fopr_item.font().underline()

        assert rft_item.text() == "\u26a0 Active"
        assert rft_item.font().underline()
