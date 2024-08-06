from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QComboBox, QWidget

from ert.gui.simulation.experiment_panel import ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.run_models import MultipleDataAssimilation

from .conftest import get_child


def test_restart_esmda(ensemble_experiment_has_run_no_failure, qtbot):
    """This runs an es-mda run from an ensemble created from an ensemble_experiment
    run, via the restart feature for es-mda.
    Regression test for several issues where this failed only in the gui.
    """
    gui = ensemble_experiment_has_run_no_failure

    experiment_panel = get_child(gui, ExperimentPanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(MultipleDataAssimilation.name())

    es_mda_panel = gui.findChild(QWidget, name="ES_MDA_panel")
    assert es_mda_panel
    restart_checkbox = es_mda_panel.findChild(QCheckBox, name="restart_checkbox_esmda")
    assert restart_checkbox
    restart_checkbox.click()
    assert restart_checkbox.isChecked()

    es_mda_panel._ensemble_selector.setCurrentText("iter-0")
    assert es_mda_panel._ensemble_selector.selected_ensemble.name == "iter-0"
    run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
    run_dialog = gui.findChild(RunDialog)
    qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=60000)
    assert (
        run_dialog._total_progress_label.text()
        == "Total progress 100% — Experiment completed."
    )

    qtbot.mouseClick(run_dialog.done_button, Qt.MouseButton.LeftButton)
