import contextlib
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QComboBox, QMessageBox, QWidget

from ert.gui.ertwidgets.listeditbox import ListEditBox
from ert.gui.ertwidgets.pathchooser import PathChooser
from ert.gui.simulation.experiment_panel import EnsembleExperimentPanel, ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.tools.export.export_panel import ExportDialog
from ert.libres_facade import LibresFacade
from ert.run_models import EnsembleExperiment
from ert.storage import open_storage

from .conftest import get_child, wait_for_child


def export_data(gui, qtbot, ensemble_select):
    file_name = None
    export_path = "output.csv"

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
            finished_message.button(QMessageBox.StandardButton.Ok),
            Qt.MouseButton.LeftButton,
        )

    QTimer.singleShot(500, handle_export_dialog)
    QTimer.singleShot(3000, handle_finished_box)
    gui.export_tool.trigger()

    assert file_name == export_path
    qtbot.waitUntil(lambda: os.path.exists(file_name))

    return file_name


def verify_exported_content(file_name, gui, ensemble_select):
    file_content = Path(file_name).read_text(encoding="utf-8")
    ensemble_names = [ensemble_select]
    if ensemble_select == "*":
        ensemble_names = [e.name for e in gui.notifier.storage.ensembles]
    for name in ensemble_names:
        experiment = gui.notifier.storage.get_experiment_by_name("es_mda")
        ensemble = experiment.get_ensemble_by_name(name)
        gen_kw_data = ensemble.load_scalars()

        facade = LibresFacade.from_config_file("poly.ert")
        misfit_data = facade.load_all_misfit_data(ensemble)

        for i in range(ensemble.ensemble_size):
            row = gen_kw_data.row(i, named=True)
            assert (
                f",{name},{row['COEFFS:a']:.6f},{row['COEFFS:b']:.6f},{row['COEFFS:c']:.6f},{misfit_data.iloc[i]['MISFIT:POLY_OBS']:.6f},{misfit_data.iloc[i]['MISFIT:TOTAL']:.6f}"
                in file_content
            )


@pytest.mark.parametrize("ensemble_select", ["default_0", "*"])
def test_csv_export(esmda_has_run, qtbot, ensemble_select):
    gui = esmda_has_run

    file_name = export_data(gui, qtbot, ensemble_select)
    verify_exported_content(file_name, gui, ensemble_select)


def run_experiment_via_gui(gui, qtbot):
    experiment_panel = get_child(gui, ExperimentPanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(EnsembleExperiment.display_name())
    ensemble_experiment_panel = get_child(experiment_panel, EnsembleExperimentPanel)
    ensemble_experiment_panel._ensemble_name_field.setText("iter-0")

    # Avoids run path dialog
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree("poly_out")

    run_experiment = get_child(experiment_panel, QWidget, name="run_experiment")
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)

    run_dialog = wait_for_child(gui, qtbot, RunDialog)
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=20000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)


def test_that_export_tool_does_not_produce_duplicate_data(
    ensemble_experiment_has_run_no_failure, qtbot
):
    """
    Ensures the export tool does not produce duplicate data by comparing
    the first and second halves of the exported CSV.

    This test addresses a previous issue where running two experiments with
    ensembles of the same name caused duplicated data in the export. The code
    has been fixed to prevent this, ensuring unique data even with identical
    ensemble names across different experiments.
    """
    gui = ensemble_experiment_has_run_no_failure

    run_experiment_via_gui(gui, qtbot)

    file_name = export_data(gui, qtbot, "*")

    df = pd.read_csv(file_name)

    # Make sure there are two identically named ensembles in two
    # different experiments.
    with open_storage("storage") as storage:
        experiments = [exp.name for exp in storage.experiments]
        ensembles = [ens.name for ens in storage.ensembles]
        assert sorted(experiments) == sorted(
            ["ensemble_experiment_0", "ensemble_experiment"]
        )
        assert ensembles == ["iter-0", "iter-0"]

    # Split the dataframe into two halves
    half_point = len(df) // 2
    first_half = df.iloc[:half_point]
    second_half = df.iloc[half_point:]

    # Ensure the two halves are not identical
    assert not first_half.equals(second_half), (
        "The first half of the data is identical to the second half."
    )
