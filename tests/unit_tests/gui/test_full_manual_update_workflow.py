import shutil

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QApplication, QComboBox, QMessageBox, QPushButton, QWidget

from ert._c_wrappers.config.rangestring import rangestring_to_mask
from ert.gui.ertwidgets.caselist import CaseList
from ert.gui.simulation.ensemble_experiment_panel import EnsembleExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.simulation_panel import SimulationPanel
from ert.shared.models import EnsembleExperiment

from .conftest import find_cases_dialog_and_panel


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
        assert case_selector.currentText().startswith("iter-0")

        # Click on "Run" and click ok on the message box
        def handle_dialog():
            messagebox = QApplication.activeWindow()
            assert isinstance(messagebox, QMessageBox)
            ok_button = messagebox.button(QMessageBox.Ok)
            qtbot.mouseClick(ok_button, Qt.LeftButton)

        QTimer.singleShot(1000, handle_dialog)
        qtbot.mouseClick(
            dialog.findChild(QPushButton, name="RUN"),
            Qt.LeftButton,
        )

    QTimer.singleShot(2000, handle_analysis_dialog)
    analysis_tool.trigger()

    # Open the manage cases dialog
    def handle_manage_dialog():
        dialog, cases_panel = find_cases_dialog_and_panel(gui, qtbot)

        # In the "create new case" tab, it should now contain "iter-1"
        cases_panel.setCurrentIndex(0)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "create_new_case_tab"
        case_list = current_tab.findChild(CaseList)
        assert isinstance(case_list, CaseList)
        assert len(case_list._list.findItems("iter-1", Qt.MatchFlag.MatchContains)) == 1
        dialog.close()

    QTimer.singleShot(1000, handle_manage_dialog)
    manage_tool = gui.tools["Manage cases"]
    manage_tool.trigger()

    # Select correct experiment in the simulation panel
    simulation_panel = gui.findChild(SimulationPanel)
    simulation_mode_combo = simulation_panel.findChild(QComboBox)
    simulation_settings = simulation_panel.findChild(EnsembleExperimentPanel)
    simulation_mode_combo.setCurrentText(EnsembleExperiment.name())
    shutil.rmtree("poly_out")

    current_select = 0
    simulation_settings._case_selector.setCurrentIndex(current_select)
    while simulation_settings._case_selector.currentText() != "iter-0":
        current_select += 1
        simulation_settings._case_selector.setCurrentIndex(current_select)

    active_reals = rangestring_to_mask(
        simulation_panel.getSimulationArguments().realizations,
        analysis_tool.ert.getEnsembleSize(),
    )
    current_select = 0
    simulation_settings._case_selector.setCurrentIndex(current_select)
    while simulation_settings._case_selector.currentText() != "iter-1":
        current_select += 1
        simulation_settings._case_selector.setCurrentIndex(current_select)

    # Assert that some realizations failed
    assert len(
        [
            r
            for r in rangestring_to_mask(
                simulation_panel.getSimulationArguments().realizations,
                analysis_tool.ert.getEnsembleSize(),
            )
            if r
        ]
    ) < len([r for r in active_reals if r])

    # Click start simulation and agree to the message
    start_simulation = simulation_panel.findChild(QWidget, name="start_simulation")

    def handle_dialog():
        message_box = gui.findChild(QMessageBox)
        qtbot.mouseClick(message_box.buttons()[0], Qt.LeftButton)

    QTimer.singleShot(500, handle_dialog)

    # The Run dialog opens, click show details and wait until done appears
    # then click it
    def use_rundialog():
        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
        run_dialog = gui.findChild(RunDialog)

        qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)

        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
        qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

        qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)

    QTimer.singleShot(1000, use_rundialog)
    qtbot.mouseClick(start_simulation, Qt.LeftButton)

    facade = simulation_panel.facade
    storage = gui.notifier.storage
    df_prior = facade.load_all_gen_kw_data(storage.get_ensemble_by_name("iter-0"))
    df_posterior = facade.load_all_gen_kw_data(storage.get_ensemble_by_name("iter-1"))

    # We expect that ERT's update step lowers the
    # generalized variance for the parameters.
    assert (
        0
        < np.linalg.det(df_posterior.cov().to_numpy())
        < np.linalg.det(df_prior.cov().to_numpy())
    )
