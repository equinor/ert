import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QPushButton,
    QWidget,
)

from ert.config import LocalizationType
from ert.gui.ertwidgets import StringBox
from ert.gui.experiments import ExperimentPanel, RunDialog
from ert.run_models import EnsembleExperiment, MultipleDataAssimilation
from ert.storage import open_storage

from .conftest import get_child


@pytest.mark.parametrize("edit_localization", [False, True])
def test_that_running_from_a_prior_persists_only_run_local_parameter_edits(
    ensemble_experiment_has_run_no_failure, qtbot, edit_localization
):
    """This runs an es-mda run from an ensemble created from an ensemble_experiment
    run, via the run prior feature for es-mda.
    Regression test for several issues where this failed only in the gui.
    """
    gui = ensemble_experiment_has_run_no_failure

    experiment_panel = get_child(gui, ExperimentPanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(MultipleDataAssimilation.display_name())

    es_mda_panel = gui.findChild(QWidget, name="ES_MDA_panel")
    assert es_mda_panel
    select_prior_checkbox = es_mda_panel.findChild(
        QCheckBox, name="select_prior_checkbox_esmda"
    )
    assert select_prior_checkbox
    select_prior_checkbox.click()
    assert select_prior_checkbox.isChecked()

    es_mda_panel._ensemble_selector.setCurrentText("iter-0")
    assert es_mda_panel._ensemble_selector.selected_ensemble.name == "iter-0"
    prior = es_mda_panel._ensemble_selector.selected_ensemble
    original_parameters = {
        name: parameter.model_dump(mode="json")
        for name, parameter in prior.experiment.parameter_configuration.items()
    }
    prior_experiment_id = prior.experiment.id
    if edit_localization:

        def edit_parameters():
            dialog = QApplication.activeModalWidget()
            assert isinstance(dialog, QDialog)
            selector = dialog.findChildren(QComboBox)[0]
            selector.setCurrentIndex(selector.findData(LocalizationType.ADAPTIVE))
            dialog.accept()

        QTimer.singleShot(0, edit_parameters)
        es_mda_panel.findChild(QPushButton, "analysis_variables_popup_button").click()
        assert all(
            parameter.update_strategy == LocalizationType.ADAPTIVE
            for parameter in es_mda_panel.active_parameters
        )
    assert original_parameters == {
        name: parameter.model_dump(mode="json")
        for name, parameter in prior.experiment.parameter_configuration.items()
    }
    run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
    run_dialog = gui.findChild(RunDialog)
    qtbot.waitUntil(lambda: run_dialog.is_experiment_done() is True, timeout=60000)
    assert (
        run_dialog._total_progress_label.text()
        == "Total progress 100% — Experiment completed."
    )
    with open_storage(gui.notifier.storage.path) as storage:
        assert original_parameters == {
            name: parameter.model_dump(mode="json")
            for name, parameter in storage.get_experiment(
                prior_experiment_id
            ).parameter_configuration.items()
        }
        target = storage.get_experiment_by_name("Run from iter-0")
        assert all(
            parameter.update_strategy
            == (
                LocalizationType.ADAPTIVE
                if edit_localization
                else LocalizationType.GLOBAL
            )
            for parameter in target.parameter_configuration.values()
        )


def test_that_esmda_active_realizations_are_set_only_when_select_prior_is_checked(
    opened_main_window_poly, qtbot
):
    """This runs a experiment and then verifies that this does
    not interfere with the activate realizations in the es_mda panel unless the
    select_prior from that specific ensemble is checked.
    """
    gui = opened_main_window_poly

    experiment_panel = get_child(gui, ExperimentPanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(EnsembleExperiment.display_name())

    ensemble_run_panel = gui.findChild(QWidget, name="Ensemble_experiment_panel")
    assert ensemble_run_panel
    ensemble_run_panel._active_realizations_field.setText("0-1")

    run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
    run_dialog = gui.findChild(RunDialog)
    qtbot.waitUntil(lambda: run_dialog.is_experiment_done() is True, timeout=15000)
    assert (
        run_dialog._total_progress_label.text()
        == "Total progress 100% — Experiment completed."
    )

    simulation_mode_combo.setCurrentText(MultipleDataAssimilation.display_name())
    es_mda_panel = gui.findChild(QWidget, name="ES_MDA_panel")
    assert es_mda_panel
    active_reals = es_mda_panel.findChild(StringBox, "active_realizations_box")
    assert active_reals.text() == "0-9"

    select_prior_checkbox = es_mda_panel.findChild(
        QCheckBox, name="select_prior_checkbox_esmda"
    )
    assert select_prior_checkbox
    assert not select_prior_checkbox.isChecked()
    select_prior_checkbox.click()
    assert active_reals.text() == "0-1"
    select_prior_checkbox.click()
    assert active_reals.text() == "0-9"


def test_custom_weights_stored_and_retrieved_from_metadata_esmda(
    opened_main_window_minimal_realizations, qtbot
):
    """This tests verifies that weights are stored in the metadata.json file
    when running esmda and that the content is read back and populated in the
    GUI when enabling selecting prior ensemble functionality.
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
    qtbot.waitUntil(lambda: run_dialog.is_experiment_done() is True, timeout=20000)
    assert (
        run_dialog._total_progress_label.text()
        == "Total progress 100% — Experiment completed."
    )
    assert wsb.text() == default_weights
    select_prior_checkbox = es_mda_panel.findChild(
        QCheckBox, name="select_prior_checkbox_esmda"
    )
    assert select_prior_checkbox
    assert not select_prior_checkbox.isChecked()
    select_prior_checkbox.click()
    # selecting prior ensemble will trigger
    # reading of metadata.json containing custom weights
    assert select_prior_checkbox.isChecked()
    assert not wsb.isEnabled()
    assert wsb.text() == custom_weights
