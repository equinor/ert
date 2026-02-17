from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QComboBox, QWidget

from ert.gui.ertwidgets import StringBox
from ert.gui.experiments.experiment_panel import ExperimentPanel
from ert.gui.experiments.run_dialog import RunDialog
from ert.run_models import MultipleDataAssimilation, SingleTestRun

from .conftest import get_child


def test_restart_esmda(ensemble_experiment_has_run_no_failure, qtbot):
    """This runs an es-mda run from an ensemble created from an ensemble_experiment
    run, via the restart feature for es-mda.
    Regression test for several issues where this failed only in the gui.
    """
    gui = ensemble_experiment_has_run_no_failure

    experiment_panel = get_child(gui, ExperimentPanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(MultipleDataAssimilation.display_name())

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
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=60000)
    assert (
        run_dialog._total_progress_label.text()
        == "Total progress 100% — Experiment completed."
    )


def test_active_realizations_esmda(opened_main_window_poly, qtbot):
    """This runs a single test run and then verifies that this does
    not interfere with the activate realizations in the es_mda panel
    unless the restart from that specific ensemble is checked.
    """
    gui = opened_main_window_poly

    experiment_panel = get_child(gui, ExperimentPanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(SingleTestRun.display_name())

    single_test_run_panel = gui.findChild(QWidget, name="Single_test_run_panel")
    assert single_test_run_panel
    run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
    run_dialog = gui.findChild(RunDialog)
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=15000)
    assert (
        run_dialog._total_progress_label.text()
        == "Total progress 100% — Experiment completed."
    )

    simulation_mode_combo.setCurrentText(MultipleDataAssimilation.display_name())
    es_mda_panel = gui.findChild(QWidget, name="ES_MDA_panel")
    assert es_mda_panel
    active_reals = es_mda_panel.findChild(StringBox, "active_realizations_box")
    assert active_reals.text() == "0-9"

    restart_checkbox = es_mda_panel.findChild(QCheckBox, name="restart_checkbox_esmda")
    assert restart_checkbox
    assert not restart_checkbox.isChecked()
    restart_checkbox.click()
    assert active_reals.text() == "0"
    restart_checkbox.click()
    assert active_reals.text() == "0-9"


def test_custom_weights_stored_and_retrieved_from_metadata_esmda(
    opened_main_window_minimal_realizations, qtbot
):
    """This tests verifies that weights are stored in the metadata.json file
    when running esmda and that the content is read back and populated in the
    GUI when enabling restart functionality.
    """
    gui = opened_main_window_minimal_realizations

    experiment_panel = get_child(gui, ExperimentPanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(MultipleDataAssimilation.display_name())

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
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=20000)
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
