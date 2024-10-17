from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QComboBox, QWidget

from ert.gui.ertwidgets import StringBox
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
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() == True, timeout=60000)
    assert (
        run_dialog._total_progress_label.text()
        == "Total progress 100% — Experiment completed."
    )


def test_custom_weights_stored_and_retrieved_from_metadata_esmda(
    opened_main_window_minimal_realizations, qtbot
):
    """This tests verifies that weights are stored in the metadata.json file when running esmda
    and that the content is read back and populated in the GUI when enabling restart functionality.
    """
    gui = opened_main_window_minimal_realizations

    experiment_panel = get_child(gui, ExperimentPanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(MultipleDataAssimilation.name())

    es_mda_panel = gui.findChild(QWidget, name="ES_MDA_panel")
    assert es_mda_panel

    custom_weights = "5, 4, 3"
    default_weights = "4, 2, 1"

    wsb = gui.findChild(StringBox, "weights_input_esmda")
    assert wsb
    assert wsb.isEnabled()
    assert wsb.text() == default_weights
    wsb.setText(custom_weights)

    # run es_mda
    run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None, timeout=5000)
    run_dialog = gui.findChild(RunDialog)
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() == True, timeout=20000)
    assert (
        run_dialog._total_progress_label.text()
        == "Total progress 100% — Experiment completed."
    )
    assert wsb.text() == default_weights
    restart_checkbox = es_mda_panel.findChild(QCheckBox, name="restart_checkbox_esmda")
    assert restart_checkbox
    assert not restart_checkbox.isChecked()
    restart_checkbox.click()
    # selecting restart will trigger reading of metadata.json containing custom weights
    assert restart_checkbox.isChecked()
    assert not wsb.isEnabled()
    assert wsb.text() == custom_weights
