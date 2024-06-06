import contextlib
import shutil

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QMessageBox,
    QPushButton,
    QTreeView,
    QWidget,
)

from ert.data import MeasuredData
from ert.gui.ertwidgets.storage_widget import StorageWidget
from ert.gui.simulation.evaluate_ensemble_panel import EvaluateEnsemblePanel
from ert.gui.simulation.experiment_panel import ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.tools.manage_experiments.manage_experiments_tool import (
    ManageExperimentsTool,
)
from ert.run_models.evaluate_ensemble import EvaluateEnsemble
from ert.validation import rangestring_to_mask

from .conftest import get_child, wait_for_child


def test_that_the_manual_analysis_tool_works(ensemble_experiment_has_run, qtbot):
    """This runs a full manual update workflow, first running ensemble experiment
    where some of the realizations fail, then doing an update before running an
    ensemble experiment again to calculate the forecast of the update.
    """
    gui = ensemble_experiment_has_run
    analysis_tool = gui.tools["Run analysis"]

    # Select correct experiment in the simulation panel
    experiment_panel = get_child(gui, ExperimentPanel)
    simulation_settings = get_child(experiment_panel, EvaluateEnsemblePanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(EvaluateEnsemble.name())

    # Open the "Run analysis" tool in the main window after ensemble experiment has run
    def handle_analysis_dialog():
        dialog = analysis_tool._dialog

        # Set target case to "iter-1"
        run_panel = analysis_tool._run_widget
        run_panel.target_ensemble_text.setText("iter-1")

        # Source case is "iter-0"
        ensemble_selector = run_panel.source_ensemble_selector
        idx = ensemble_selector.findData("iter-0", Qt.MatchStartsWith)
        assert idx != -1
        ensemble_selector.setCurrentIndex(idx)

        # Click on "Run" and click ok on the message box
        def handle_dialog():
            qtbot.waitUntil(
                lambda: isinstance(QApplication.activeWindow(), QMessageBox)
            )
            messagebox = QApplication.activeWindow()
            assert isinstance(messagebox, QMessageBox)
            ok_button = messagebox.button(QMessageBox.Ok)
            qtbot.mouseClick(ok_button, Qt.LeftButton)

        QTimer.singleShot(1000, handle_dialog)
        qtbot.mouseClick(
            get_child(dialog, QPushButton, name="RUN"),
            Qt.LeftButton,
        )

    QTimer.singleShot(2000, handle_analysis_dialog)
    analysis_tool.trigger()

    # Open the manage experiments dialog
    manage_tool = gui.tools["Manage experiments"]
    manage_tool.trigger()

    assert isinstance(manage_tool, ManageExperimentsTool)
    experiments_panel = manage_tool._ensemble_management_widget

    # In the "create new case" tab, it should now contain "iter-1"
    experiments_panel.setCurrentIndex(0)
    current_tab = experiments_panel.currentWidget()
    assert current_tab.objectName() == "create_new_ensemble_tab"
    storage_widget = get_child(current_tab, StorageWidget)
    tree_view = get_child(storage_widget, QTreeView)
    tree_view.expandAll()

    model = tree_view.model()
    assert model is not None and model.rowCount() == 2
    assert "iter-1" in model.index(1, 0, model.index(1, 0)).data(0)

    experiments_panel.close()

    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree("poly_out")

    idx = simulation_settings._ensemble_selector.findData("iter-1", Qt.MatchStartsWith)
    assert idx != -1
    simulation_settings._ensemble_selector.setCurrentIndex(idx)

    storage = gui.notifier.storage
    ensemble_prior = storage.get_ensemble_by_name("iter-0")
    active_reals = list(ensemble_prior.get_realization_mask_with_responses())
    # Assert that some realizations failed
    assert not all(active_reals)
    assert active_reals == rangestring_to_mask(
        experiment_panel.get_experiment_arguments().realizations,
        analysis_tool.ert.ert_config.model_config.num_realizations,
    )
    # Click start simulation and agree to the message
    run_experiment = get_child(experiment_panel, QWidget, name="run_experiment")

    qtbot.mouseClick(run_experiment, Qt.LeftButton)
    # The Run dialog opens, click show details and wait until done appears
    # then click it
    run_dialog = wait_for_child(gui, qtbot, RunDialog)
    qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)
    qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)
    qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)

    df_prior = ensemble_prior.load_all_gen_kw_data()
    ensemble_posterior = storage.get_ensemble_by_name("iter-1")
    df_posterior = ensemble_posterior.load_all_gen_kw_data()

    # Making sure measured data works with failed realizations
    MeasuredData(storage.get_ensemble_by_name("iter-0"), ["POLY_OBS"])

    # We expect that ERT's update step lowers the
    # generalized variance for the parameters.
    assert (
        0
        < np.linalg.det(df_posterior.cov().to_numpy())
        < np.linalg.det(df_prior.cov().to_numpy())
    )
