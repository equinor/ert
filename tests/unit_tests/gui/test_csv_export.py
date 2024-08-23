import contextlib
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QComboBox, QMessageBox, QWidget

from ert.gui.ertwidgets.listeditbox import ListEditBox
from ert.gui.ertwidgets.pathchooser import PathChooser
from ert.gui.simulation.experiment_panel import EnsembleExperimentPanel, ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.tools.export.export_panel import ExportDialog
from ert.libres_facade import LibresFacade
from ert.run_models import EnsembleExperiment

from .conftest import get_child, wait_for_child


def export_data(gui, qtbot, ensemble_select, export_path="output.csv"):
    file_name = None

    def handle_export_dialog():
        nonlocal file_name

        # Find the case selection box in the dialog
        export_dialog = wait_for_child(gui, qtbot, ExportDialog)
        case_selection = get_child(export_dialog, ListEditBox)

        # Select ensemble_select as the case to be exported
        case_selection._list_edit_line.setText(ensemble_select)
        path_chooser = get_child(export_dialog, PathChooser)
        file_name = path_chooser._path_line.text()
        assert case_selection.isValid()

        qtbot.mouseClick(export_dialog.ok_button, Qt.MouseButton.LeftButton)

    def handle_finished_box():
        """
        Click on the plugin finished dialog once it pops up
        """
        finished_message = wait_for_child(gui, qtbot, QMessageBox)
        assert "completed" in finished_message.text()
        qtbot.mouseClick(
            finished_message.button(QMessageBox.Ok), Qt.MouseButton.LeftButton
        )

    QTimer.singleShot(500, handle_export_dialog)
    QTimer.singleShot(3000, handle_finished_box)

    gui.tools["Export data"].trigger()
    assert file_name == export_path
    qtbot.waitUntil(lambda: os.path.exists(file_name))

    return file_name


def verify_exported_content(file_name, gui, ensemble_select):
    file_content = Path(file_name).read_text(encoding="utf-8")
    ensemble_names = [ensemble_select]
    if ensemble_select == "*":
        ensemble_names = [e.name for e in gui.notifier.storage.ensembles]
    for name in ensemble_names:
        ensemble = gui.notifier.storage.get_ensemble_by_name(name)
        gen_kw_data = ensemble.load_all_gen_kw_data()

        facade = LibresFacade.from_config_file("poly.ert")
        misfit_data = facade.load_all_misfit_data(ensemble)

        for i in range(ensemble.ensemble_size):
            assert (
                f",{name},{gen_kw_data.iloc[i]['COEFFS:a']},{gen_kw_data.iloc[i]['COEFFS:b']},{gen_kw_data.iloc[i]['COEFFS:c']},{misfit_data.iloc[i]['MISFIT:POLY_OBS']},{misfit_data.iloc[i]['MISFIT:TOTAL']}"
                in file_content
            )


@pytest.mark.parametrize("ensemble_select", ["default_0", "*"])
def test_csv_export(esmda_has_run, qtbot, ensemble_select):
    gui = esmda_has_run

    file_name = export_data(gui, qtbot, ensemble_select)
    verify_exported_content(file_name, gui, ensemble_select)


def run_experiment_and_export(gui, qtbot):
    experiment_panel = get_child(gui, ExperimentPanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(EnsembleExperiment.name())
    ensemble_experiment_panel = get_child(experiment_panel, EnsembleExperimentPanel)
    ensemble_experiment_panel._ensemble_name_field.setText("iter-0")

    # Avoids run path dialog
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree("poly_out")

    run_experiment = get_child(experiment_panel, QWidget, name="run_experiment")
    qtbot.mouseClick(run_experiment, Qt.LeftButton)

    run_dialog = wait_for_child(gui, qtbot, RunDialog)
    qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)
    qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)


def test_that_export_tool_does_not_produce_duplicate_data(
    ensemble_experiment_has_run_no_failure, qtbot
):
    gui = ensemble_experiment_has_run_no_failure

    run_experiment_and_export(gui, qtbot)

    file_name = export_data(gui, qtbot, "*")

    df = pd.read_csv(file_name)
    # Make sure data is not duplicated.
    assert df.iloc[0]["COEFFS:a"] != df.iloc[20]["COEFFS:a"]
