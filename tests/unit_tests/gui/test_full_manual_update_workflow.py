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
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.simulation_panel import SimulationPanel
from ert.run_models.evaluate_ensemble import EvaluateEnsemble
from ert.validation import rangestring_to_mask

from .conftest import get_child, wait_for_child, with_manage_tool


def test_that_the_manual_analysis_tool_works(
    ensemble_experiment_has_run, opened_main_window, qtbot, run_experiment
):
    """This runs a full manual update workflow, first running ensemble experiment
    where some of the realizations fail, then doing an update before running an
    ensemble experiment again to calculate the forecast of the update.
    """
    gui = opened_main_window
    analysis_tool = gui.tools["Run analysis"]

    # Open the "Run analysis" tool in the main window after ensemble experiment has run
    def handle_analysis_dialog():
        dialog = analysis_tool._dialog

        # Set target case to "iter-1"
        run_panel = analysis_tool._run_widget
        run_panel.target_case_text.setText("iter-1")

        # Source case is "iter-0"
        case_selector = run_panel.source_case_selector
        current_select = 0
        case_selector.setCurrentIndex(current_select)
        while case_selector.currentText() != "iter-0":
            current_select += 1
            simulation_settings._case_selector.setCurrentIndex(current_select)
        assert case_selector.currentText().startswith("iter-0")

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

    # Open the manage cases dialog
    def handle_manage_dialog(dialog, cases_panel):
        # In the "create new case" tab, it should now contain "iter-1"
        cases_panel.setCurrentIndex(0)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "create_new_case_tab"
        storage_widget = get_child(current_tab, StorageWidget)
        tree_view = get_child(storage_widget, QTreeView)
        tree_view.expandAll()

        assert tree_view.model().rowCount() == 2
        assert "iter-1" in tree_view.model().index(
            1, 0, tree_view.model().index(1, 0)
        ).data(0)

        dialog.close()

    with_manage_tool(gui, qtbot, handle_manage_dialog)

    # Select correct experiment in the simulation panel
    simulation_panel = get_child(gui, SimulationPanel)
    simulation_mode_combo = get_child(simulation_panel, QComboBox)
    simulation_settings = get_child(simulation_panel, EvaluateEnsemblePanel)
    simulation_mode_combo.setCurrentText(EvaluateEnsemble.name())

    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree("poly_out")

    current_select = 0
    simulation_settings._case_selector.setCurrentIndex(current_select)
    while simulation_settings._case_selector.currentText() != "iter-1":
        current_select += 1
        simulation_settings._case_selector.setCurrentIndex(current_select)

    storage = gui.notifier.storage
    ensemble_prior = storage.get_ensemble_by_name("iter-0")
    active_reals = list(ensemble_prior.get_realization_mask_with_responses())
    # Assert that some realizations failed
    assert not all(active_reals)
    assert active_reals == rangestring_to_mask(
        simulation_panel.getSimulationArguments().realizations,
        analysis_tool.ert.ert_config.model_config.num_realizations,
    )
    # Click start simulation and agree to the message
    start_simulation = get_child(simulation_panel, QWidget, name="start_simulation")

    qtbot.mouseClick(start_simulation, Qt.LeftButton)
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
