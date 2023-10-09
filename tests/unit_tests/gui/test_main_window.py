import contextlib
import os
import shutil
import stat
from textwrap import dedent
from typing import Optional

import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QToolButton,
    QWidget,
)

from ert.gui.ertwidgets import ClosableDialog
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel
from ert.gui.ertwidgets.caselist import AddRemoveWidget, CaseList
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.customdialog import CustomDialog
from ert.gui.ertwidgets.listeditbox import ListEditBox
from ert.gui.ertwidgets.pathchooser import PathChooser
from ert.gui.ertwidgets.validateddialog import ValidatedDialog
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.simulation_panel import SimulationPanel
from ert.gui.tools.manage_cases.case_init_configuration import (
    CaseInitializationConfigurationPanel,
)
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_case_selection_widget import CaseSelectionWidget
from ert.gui.tools.plot.plot_window import PlotWindow
from ert.run_models import SingleTestRun
from tests.unit_tests.gui.simulation.test_run_path_dialog import handle_run_path_dialog

from .conftest import find_cases_dialog_and_panel, load_results_manually


@pytest.mark.usefixtures("use_tmpdir")
def test_that_the_plot_window_contains_the_expected_elements(
    esmda_has_run, opened_main_window, qtbot
):
    gui = opened_main_window
    expected_cases = [
        "default",
        "default_0",
        "default_1",
        "default_2",
        "default_3",
    ]

    # Click on Create plot after esmda has run
    plot_tool = gui.tools["Create plot"]
    plot_tool.trigger()

    # Then the plot window opens
    qtbot.waitUntil(lambda: gui.findChild(PlotWindow) is not None)
    plot_window = gui.findChild(PlotWindow)
    assert isinstance(plot_window, PlotWindow)

    case_selection = plot_window.findChild(CaseSelectionWidget)
    data_types = plot_window.findChild(DataTypeKeysWidget)
    assert isinstance(data_types, DataTypeKeysWidget)
    combo_box: Optional[QComboBox] = case_selection.findChild(
        QComboBox, "case_selector"
    )  # type: ignore
    assert combo_box is not None

    # Assert that the Case selection widget contains the expected cases
    case_names = []
    for i in range(combo_box.count()):
        case_names.append(combo_box.itemText(i))
    assert sorted(case_names) == expected_cases

    data_names = []
    data_keys = data_types.data_type_keys_widget
    for i in range(data_keys.model().rowCount()):
        index = data_keys.model().index(i, 0)
        data_names.append(str(index.data(Qt.DisplayRole)))
    assert data_names == [
        "POLY_RES@0",
        "COEFFS:COEFF_A",
        "COEFFS:COEFF_B",
        "COEFFS:COEFF_C",
    ]

    assert {
        plot_window._central_tab.tabText(i)
        for i in range(plot_window._central_tab.count())
    } == {
        "Cross case statistics",
        "Distribution",
        "Gaussian KDE",
        "Ensemble",
        "Histogram",
        "Statistics",
    }

    # Add cases to plot
    for _ in expected_cases:
        qtbot.mouseClick(
            case_selection.findChild(QToolButton, name="add_case_button"), Qt.LeftButton
        )
    all_added_combo_boxes = case_selection.findChildren(QComboBox, name="case_selector")
    assert len(expected_cases) == len(all_added_combo_boxes)

    # make sure pairwise different and all cases are selected
    for case_index, case_name in enumerate(expected_cases):
        combo_box = all_added_combo_boxes[case_index]
        for i in range(combo_box.count()):
            if case_name == combo_box.itemText(i):
                combo_box.setCurrentIndex(i)
    assert {box.currentText() for box in all_added_combo_boxes} == set(expected_cases)

    # Cycle through showing all the tabs and plot each data key

    for i in range(data_keys.model().rowCount()):
        index = data_keys.model().index(i, 0)
        qtbot.mouseClick(
            data_types.data_type_keys_widget,
            Qt.LeftButton,
            pos=data_types.data_type_keys_widget.visualRect(index).center(),
        )
        for tab_index in range(plot_window._central_tab.count()):
            if not plot_window._central_tab.isTabEnabled(tab_index):
                continue
            plot_window._central_tab.setCurrentIndex(tab_index)
    plot_window.close()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_the_manage_cases_tool_can_be_used(
    esmda_has_run,
    opened_main_window,
    qtbot,
):
    gui = opened_main_window

    # Click on "Manage Cases"
    def handle_dialog():
        dialog, cases_panel = find_cases_dialog_and_panel(gui, qtbot)
        # Open the create new cases tab
        cases_panel.setCurrentIndex(0)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "create_new_case_tab"
        create_widget = current_tab.findChild(AddRemoveWidget)
        case_list = current_tab.findChild(CaseList)
        assert isinstance(case_list, CaseList)

        # The case list should contain the expected cases
        assert case_list._list.count() == 5

        # Click add case and name it "new_case"
        def handle_add_dialog():
            qtbot.waitUntil(lambda: current_tab.findChild(ValidatedDialog) is not None)
            dialog = gui.findChild(ValidatedDialog)
            dialog.param_name.setText("new_case")
            qtbot.mouseClick(dialog.ok_button, Qt.LeftButton)

        QTimer.singleShot(1000, handle_add_dialog)
        qtbot.mouseClick(create_widget.addButton, Qt.LeftButton)

        # The list should now contain "new_case"
        assert case_list._list.count() == 6

        # Click add case and try to name it "new_case" again
        def handle_add_dialog_again():
            qtbot.waitUntil(lambda: current_tab.findChild(ValidatedDialog) is not None)
            dialog = gui.findChild(ValidatedDialog)
            dialog.param_name.setText("new_case")
            assert not dialog.ok_button.isEnabled()
            qtbot.mouseClick(dialog.cancel_button, Qt.LeftButton)

        QTimer.singleShot(1000, handle_add_dialog_again)
        qtbot.mouseClick(create_widget.addButton, Qt.LeftButton)

        # The list contains the same amount of cases as before
        assert case_list._list.count() == 6

        # Go to the "initialize from scratch" panel
        cases_panel.setCurrentIndex(1)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "initialize_from_scratch_panel"
        combo_box = current_tab.findChild(CaseSelector)
        assert isinstance(combo_box, CaseSelector)

        # Select "new_case"
        current_index = 0
        while combo_box.currentText().startswith("new_case"):
            current_index += 1
            combo_box.setCurrentIndex(current_index)

        # click on "initialize"
        initialize_button = current_tab.findChild(
            QPushButton, name="initialize_from_scratch_button"
        )
        qtbot.mouseClick(initialize_button, Qt.LeftButton)

        dialog.close()

    QTimer.singleShot(1000, handle_dialog)
    manage_tool = gui.tools["Manage cases"]
    manage_tool.trigger()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_inversion_type_can_be_set_from_gui(qtbot, opened_main_window):
    gui = opened_main_window

    sim_mode = gui.findChild(QWidget, name="Simulation_mode")
    qtbot.keyClick(sim_mode, Qt.Key_Down)
    es_panel = gui.findChild(QWidget, name="ensemble_smoother_panel")
    es_edit = es_panel.findChild(QWidget, name="ensemble_smoother_edit")

    # Testing modal dialogs requires some care.
    # A helpful discussion on the topic is here:
    # https://github.com/pytest-dev/pytest-qt/issues/256
    def handle_dialog_first_time():
        var_panel = gui.findChild(AnalysisModuleVariablesPanel)
        inversion_spin_box = var_panel.findChild(QSpinBox, name="IES_INVERSION")
        assert inversion_spin_box.value() == 0
        qtbot.keyClick(inversion_spin_box, Qt.Key_Up)
        assert inversion_spin_box.value() == 1
        var_panel.parent().close()

    QTimer.singleShot(500, handle_dialog_first_time)
    qtbot.mouseClick(es_edit.findChild(QToolButton), Qt.LeftButton, delay=1)

    def handle_dialog_second_time():
        var_panel = gui.findChild(AnalysisModuleVariablesPanel)
        inversion_spin_box = var_panel.findChild(QSpinBox, name="IES_INVERSION")
        assert inversion_spin_box.value() == 1
        var_panel.parent().close()

    QTimer.singleShot(500, handle_dialog_second_time)
    qtbot.mouseClick(es_edit.findChild(QToolButton), Qt.LeftButton, delay=1)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_csv_export_plugin_generates_a_file(
    qtbot, esmda_has_run, opened_main_window
):
    gui = opened_main_window

    # Find EXPORT_CSV in the plugin menu
    plugin_tool = gui.tools["Plugins"]
    plugin_actions = plugin_tool.getAction().menu().actions()
    export_csv_action = [a for a in plugin_actions if a.text() == "CSV Export"][0]
    file_name = None

    def handle_plugin_dialog():
        nonlocal file_name

        # Find the case selection box in the dialog
        qtbot.waitUntil(lambda: isinstance(gui.findChild(CustomDialog), CustomDialog))
        export_dialog = gui.findChild(CustomDialog)
        assert isinstance(export_dialog, CustomDialog)
        case_selection = export_dialog.findChild(ListEditBox)
        assert isinstance(case_selection, ListEditBox)

        # Select default_0 as the case to be exported
        case_selection._list_edit_line.setText("default_0")
        path_chooser = export_dialog.findChild(PathChooser)
        assert isinstance(path_chooser, PathChooser)
        file_name = path_chooser._path_line.text()
        assert case_selection.isValid()

        qtbot.mouseClick(export_dialog.ok_button, Qt.LeftButton)

    def handle_finished_box():
        """
        Click on the plugin finised dialog once it pops up
        """
        qtbot.waitUntil(lambda: isinstance(gui.findChild(QMessageBox), QMessageBox))
        finished_message = gui.findChild(QMessageBox)
        assert isinstance(finished_message, QMessageBox)
        assert "completed" in finished_message.text()
        qtbot.mouseClick(finished_message.button(QMessageBox.Ok), Qt.LeftButton)

    QTimer.singleShot(500, handle_plugin_dialog)
    QTimer.singleShot(3000, handle_finished_box)
    export_csv_action.trigger()

    assert file_name == "output.csv"
    qtbot.waitUntil(lambda: os.path.exists(file_name))


def test_that_the_manage_cases_tool_can_be_used_with_clean_storage(
    opened_main_window_clean, qtbot
):
    gui = opened_main_window_clean

    # Click on "Manage Cases"
    def handle_dialog():
        qtbot.waitUntil(lambda: gui.findChild(ClosableDialog) is not None)
        dialog = gui.findChild(ClosableDialog)
        cases_panel = dialog.findChild(CaseInitializationConfigurationPanel)
        assert isinstance(cases_panel, CaseInitializationConfigurationPanel)

        # Open the create new cases tab
        cases_panel.setCurrentIndex(0)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "create_new_case_tab"
        create_widget = current_tab.findChild(AddRemoveWidget)
        case_list = current_tab.findChild(CaseList)
        assert isinstance(case_list, CaseList)

        assert case_list._list.count() == 0

        # Click add case and name it "new_case"
        def handle_add_dialog():
            qtbot.waitUntil(lambda: current_tab.findChild(ValidatedDialog) is not None)
            dialog = gui.findChild(ValidatedDialog)
            dialog.param_name.setText("new_case")
            qtbot.mouseClick(dialog.ok_button, Qt.LeftButton)

        QTimer.singleShot(1000, handle_add_dialog)
        qtbot.mouseClick(create_widget.addButton, Qt.LeftButton)

        # The list should now contain "new_case"
        assert case_list._list.count() == 1
        assert case_list._list.item(0).data(Qt.UserRole).name == "new_case"

        # Go to the "initialize from scratch" panel
        cases_panel.setCurrentIndex(1)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "initialize_from_scratch_panel"
        combo_box = current_tab.findChild(CaseSelector)
        assert isinstance(combo_box, CaseSelector)

        assert combo_box.currentText().startswith("new_case")

        # click on "initialize"
        initialize_button = current_tab.findChild(
            QPushButton, name="initialize_from_scratch_button"
        )
        qtbot.mouseClick(initialize_button, Qt.LeftButton)

        dialog.close()

    QTimer.singleShot(1000, handle_dialog)
    manage_tool = gui.tools["Manage cases"]
    manage_tool.trigger()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_load_results_manually_can_be_run_after_esmda(
    esmda_has_run, opened_main_window, qtbot
):
    load_results_manually(qtbot, opened_main_window)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_a_failing_job_shows_error_message_with_context(
    opened_main_window_clean, qtbot
):
    gui = opened_main_window_clean

    # break poly eval script so realz fail
    with open("poly_eval.py", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """\
                #!/usr/bin/env python

                if __name__ == "__main__":
                    raise RuntimeError('Argh')
                """
            )
        )
    os.chmod(
        "poly_eval.py",
        os.stat("poly_eval.py").st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )

    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree("poly_out")
    # Select correct experiment in the simulation panel
    simulation_panel = gui.findChild(SimulationPanel)
    assert isinstance(simulation_panel, SimulationPanel)
    simulation_mode_combo = simulation_panel.findChild(QComboBox)
    assert isinstance(simulation_mode_combo, QComboBox)
    simulation_mode_combo.setCurrentText(SingleTestRun.name())

    # Click start simulation and agree to the message
    start_simulation = simulation_panel.findChild(QWidget, name="start_simulation")
    assert start_simulation
    assert isinstance(start_simulation, QToolButton)

    def handle_dialog():
        message_box = gui.findChild(QMessageBox)
        assert message_box
        qtbot.mouseClick(message_box.buttons()[0], Qt.LeftButton)

        QTimer.singleShot(
            500,
            lambda: handle_run_path_dialog(gui=gui, qtbot=qtbot, delete_run_path=False),
        )

    def handle_error_dialog(run_dialog):
        error_dialog = run_dialog.fail_msg_box
        assert error_dialog
        text = error_dialog.details_text.toPlainText()
        label = error_dialog.label_text.text()
        assert "ERT experiment failed" in label
        expected_substrings = [
            "Realization: 0 failed after reaching max submit (2)",
            "job poly_eval failed",
            "Process exited with status code 1",
            "Traceback",
            "raise RuntimeError('Argh')",
            "RuntimeError: Argh",
        ]
        for substring in expected_substrings:
            assert substring in text
        qtbot.mouseClick(error_dialog.box.buttons()[0], Qt.LeftButton)

    QTimer.singleShot(500, handle_dialog)
    qtbot.mouseClick(start_simulation, Qt.LeftButton)

    qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
    run_dialog = gui.findChild(RunDialog)
    qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)

    QTimer.singleShot(20000, lambda: handle_error_dialog(run_dialog))
    qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
