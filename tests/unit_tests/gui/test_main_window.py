import fileinput
import os.path
import shutil
import stat
from textwrap import dedent
from typing import List
from unittest.mock import Mock

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QToolButton,
    QWidget,
)

from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel
from ert.gui.ertwidgets.caselist import AddRemoveWidget, CaseList
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.ertwidgets.validateddialog import ValidatedDialog
from ert.gui.main import GUILogHandler, _setup_main_window
from ert.gui.simulation.ensemble_experiment_panel import EnsembleExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.simulation_panel import SimulationPanel
from ert.gui.simulation.view import RealizationWidget
from ert.gui.tools.manage_cases.case_init_configuration import (
    CaseInitializationConfigurationPanel,
)
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_case_selection_widget import CaseSelectionWidget
from ert.gui.tools.plot.plot_window import PlotWindow
from ert.services import StorageService
from ert.shared.models import EnsembleExperiment, MultipleDataAssimilation


@pytest.mark.usefixtures("use_tmpdir")
@pytest.fixture(scope="module")
def opened_main_window(source_root, tmpdir_factory, request):
    with pytest.MonkeyPatch.context() as mp:
        tmp_path = tmpdir_factory.mktemp("test-data")

        request.addfinalizer(lambda: shutil.rmtree(tmp_path))
        shutil.copytree(
            os.path.join(source_root, "test-data", "poly_example"),
            tmp_path / "test_data",
        )
        mp.chdir(tmp_path / "test_data")
        with fileinput.input("poly.ert", inplace=True) as fin:
            for line in fin:
                if "NUM_REALIZATIONS" in line:
                    # Decrease the number of realizations to speed up the test,
                    # if there is flakyness, this can be increased.
                    print("NUM_REALIZATIONS 20", end="\n")
                else:
                    print(line, end="")
            poly_case = EnKFMain(ErtConfig.from_file("poly.ert"))
        args_mock = Mock()
        args_mock.config = "poly.ert"

        with StorageService.init_service(
            ert_config=args_mock.config,
            project=os.path.abspath(poly_case.ert_config.ens_path),
        ):
            gui = _setup_main_window(poly_case, args_mock, GUILogHandler())
            yield gui


@pytest.mark.usefixtures("use_tmpdir, opened_main_window")
@pytest.fixture(scope="module")
def esmda_has_run(run_experiment):
    # Runs a default ES-MDA run
    run_experiment(MultipleDataAssimilation)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.fixture(scope="module")
def run_experiment(request, opened_main_window):
    def func(experiment_mode):
        qtbot = QtBot(request)
        gui = opened_main_window
        qtbot.addWidget(gui)
        try:
            shutil.rmtree("poly_out")
        except FileNotFoundError:
            pass
        # Select correct experiment in the simulation panel
        simulation_panel = gui.findChild(SimulationPanel)
        assert isinstance(simulation_panel, SimulationPanel)
        simulation_mode_combo = simulation_panel.findChild(QComboBox)
        assert isinstance(simulation_mode_combo, QComboBox)
        simulation_mode_combo.setCurrentText(experiment_mode.name())

        # Click start simulation and agree to the message
        start_simulation = simulation_panel.findChild(QWidget, name="start_simulation")

        def handle_dialog():
            message_box = gui.findChild(QMessageBox)
            qtbot.mouseClick(message_box.buttons()[0], Qt.LeftButton)

        QTimer.singleShot(500, handle_dialog)

        # The Run dialog opens, click show details and wait until done appears
        # then click it
        def use_rundialog():
            qtbot.waitUntil(lambda: isinstance(QApplication.activeWindow(), RunDialog))
            run_dialog = QApplication.activeWindow()

            qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)

            qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
            qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

            # Assert that the number of boxes in the detailed view is
            # equal to the number of realizations
            realization_widget = run_dialog._tab_widget.currentWidget()
            assert isinstance(realization_widget, RealizationWidget)
            list_model = realization_widget._real_view.model()
            assert list_model.rowCount() == simulation_panel.ert.getEnsembleSize()

            qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)

        QTimer.singleShot(1000, use_rundialog)
        qtbot.mouseClick(start_simulation, Qt.LeftButton)

    return func


@pytest.mark.usefixtures("use_tmpdir")
@pytest.fixture(scope="module")
def ensemble_experiment_has_run(opened_main_window, run_experiment, request):
    gui = opened_main_window
    qtbot = QtBot(request)

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

        # Click add case and name it "iter-0"
        def handle_add_dialog():
            qtbot.waitUntil(lambda: current_tab.findChild(ValidatedDialog) is not None)
            dialog = gui.findChild(ValidatedDialog)
            dialog.param_name.setText("iter-0")
            qtbot.mouseClick(dialog.ok_button, Qt.LeftButton)

        QTimer.singleShot(1000, handle_add_dialog)
        qtbot.mouseClick(create_widget.addButton, Qt.LeftButton)

        dialog.close()

    QTimer.singleShot(1000, handle_dialog)
    manage_tool = gui.tools["Manage cases"]
    manage_tool.trigger()

    with open("poly_eval.py", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """#!/usr/bin/env python
import numpy as np
import sys
import json

def _load_coeffs(filename):
    with open(filename, encoding="utf-8") as f:
        return json.load(f)

def _evaluate(coeffs, x):
    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]

if __name__ == "__main__":
    if np.random.random(1) > 0.5:
        sys.exit(1)
    coeffs = _load_coeffs("coeffs.json")
    output = [_evaluate(coeffs, x) for x in range(10)]
    with open("poly_0.out", "w", encoding="utf-8") as f:
        f.write("\\n".join(map(str, output)))
        """
            )
        )
    os.chmod(
        "poly_eval.py",
        os.stat("poly_eval.py").st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )
    run_experiment(EnsembleExperiment)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_the_plot_window_contains_the_expected_elements(
    esmda_has_run, opened_main_window, qtbot
):
    gui = opened_main_window

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
    combo_boxes: List[QComboBox] = case_selection.findChildren(
        QComboBox
    )  # type: ignore

    # Assert that the Case selection widget contains the expected cases
    case_names = []
    assert len(combo_boxes) == 1
    combo_box = combo_boxes[0]
    for i in range(combo_box.count()):
        data_names = []
        combo_box.setCurrentIndex(i)
        case_names.append(combo_box.currentText())
    assert case_names == [
        "default",
        "default_0",
        "default_1",
        "default_2",
        "default_3",
    ]

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

    # Add all the cases
    for name in case_names:
        combo_box = combo_boxes[-1]
        for i in range(combo_box.count()):
            if combo_box.itemText(i) == name:
                combo_box.setCurrentIndex(i)
        qtbot.mouseClick(
            case_selection.findChild(QToolButton, name="add_case_button"), Qt.LeftButton
        )
        combo_boxes: List[QComboBox] = case_selection.findChildren(
            QComboBox
        )  # type: ignore
    assert len(case_selection.findChildren(QComboBox)) == len(case_names)

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
    esmda_has_run, opened_main_window, qtbot
):
    gui = opened_main_window

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

        # Go to the "initialize from scratch" panel
        cases_panel.setCurrentIndex(1)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "initialize_from_scratch_panel"
        combo_box = current_tab.findChild(CaseSelector)
        assert isinstance(combo_box, CaseSelector)

        # Select "new_case"
        current_index = 0
        while combo_box.currentText() != "new_case":
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
        assert case_selector.currentText() == "iter-0"

        # Click on "Run" and click ok on the message box
        def handle_dialog():
            messagebox = QApplication.activeWindow()
            assert isinstance(messagebox, QMessageBox)
            ok_button = messagebox.button(QMessageBox.Ok)
            qtbot.mouseClick(ok_button, Qt.LeftButton)

        QTimer.singleShot(1000, handle_dialog)
        qtbot.mouseClick(
            dialog.findChild(QPushButton, name="Run"),
            Qt.LeftButton,
        )

    QTimer.singleShot(2000, handle_analysis_dialog)
    analysis_tool.trigger()

    # Open the manage cases dialog
    def handle_manage_dialog():
        dialog = QApplication.activeWindow()
        cases_panel = dialog.findChild(CaseInitializationConfigurationPanel)
        assert isinstance(cases_panel, CaseInitializationConfigurationPanel)

        # In the "create new case" tab, it should now contain "iter-1"
        cases_panel.setCurrentIndex(0)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "create_new_case_tab"
        case_list = current_tab.findChild(CaseList)
        assert isinstance(case_list, CaseList)
        assert len(case_list._list.findItems("iter-1", Qt.MatchFlag.MatchExactly)) == 1
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

    active_reals_string_len = len(
        simulation_panel.getSimulationArguments().realizations
    )
    current_select = 0
    simulation_settings._case_selector.setCurrentIndex(current_select)
    while simulation_settings._case_selector.currentText() != "iter-1":
        current_select += 1
        simulation_settings._case_selector.setCurrentIndex(current_select)

    # We have selected the updated case and because some realizations failed in the
    # parent ensemble we expect the active realizations string to be longer as it
    # needs to account for the missing realizations.
    assert (
        len(simulation_panel.getSimulationArguments().realizations)
        > active_reals_string_len
    )

    # Click start simulation and agree to the message
    start_simulation = simulation_panel.findChild(QWidget, name="start_simulation")

    def handle_dialog():
        message_box = gui.findChild(QMessageBox)
        qtbot.mouseClick(message_box.buttons()[0], Qt.LeftButton)

    QTimer.singleShot(500, handle_dialog)

    # The Run dialog opens, click show details and wait until done appears
    # then click it
    def use_rundialog():
        qtbot.waitUntil(lambda: isinstance(QApplication.activeWindow(), RunDialog))
        run_dialog = QApplication.activeWindow()

        qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)

        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
        qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

        qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)

    QTimer.singleShot(1000, use_rundialog)
    qtbot.mouseClick(start_simulation, Qt.LeftButton)

    facade = simulation_panel.facade
    df_prior = facade.load_all_gen_kw_data("iter-0")
    df_posterior = facade.load_all_gen_kw_data("iter-1")

    # We expect that ERT's update step lowers the
    # generalized variance for the parameters.
    assert (
        0
        < np.linalg.det(df_posterior.cov().to_numpy())
        < np.linalg.det(df_prior.cov().to_numpy())
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_inversion_type_can_be_set_from_gui(qtbot, opened_main_window):
    gui = opened_main_window
    qtbot.addWidget(gui)

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
