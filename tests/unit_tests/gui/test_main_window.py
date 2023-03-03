import os.path
import shutil
from pathlib import Path
from typing import List
from unittest.mock import Mock

import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QMessageBox,
    QPushButton,
    QToolButton,
    QWidget,
)

from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert.gui.ertwidgets.caselist import AddRemoveWidget, CaseList
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.ertwidgets.validateddialog import ValidatedDialog
from ert.gui.main import GUILogHandler, _setup_main_window
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.simulation_panel import SimulationPanel
from ert.gui.simulation.view import RealizationWidget
from ert.gui.tools.load_results.load_results_panel import LoadResultsPanel
from ert.gui.tools.manage_cases.case_init_configuration import (
    CaseInitializationConfigurationPanel,
)
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_case_selection_widget import CaseSelectionWidget
from ert.gui.tools.plot.plot_window import PlotWindow
from ert.services import StorageService
from ert.shared.models import MultipleDataAssimilation
from ert.storage import open_storage


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

        path = Path("poly.ert")
        text = path.read_text(encoding="utf-8")
        text = text.replace("NUM_REALIZATIONS 100", "NUM_REALIZATIONS 10")
        path.write_text(text, encoding="utf-8")

        poly_case = EnKFMain(ErtConfig.from_file("poly.ert"))
        args_mock = Mock()
        args_mock.config = "poly.ert"

        with StorageService.init_service(
            ert_config=args_mock.config,
            project=os.path.abspath(poly_case.ert_config.ens_path),
        ), open_storage(poly_case.ert_config.ens_path, mode="w") as storage:
            gui = _setup_main_window(poly_case, args_mock, GUILogHandler())
            gui.notifier.set_storage(storage)
            yield gui, poly_case.getEnsembleSize()


@pytest.mark.usefixtures("use_tmpdir")
@pytest.fixture(scope="function")
def opened_main_window_clean(source_root, tmpdir_factory, request):
    with pytest.MonkeyPatch.context() as mp:
        tmp_path = tmpdir_factory.mktemp("test-data")

        request.addfinalizer(lambda: shutil.rmtree(tmp_path))
        shutil.copytree(
            os.path.join(source_root, "test-data", "poly_example"),
            tmp_path / "test_data",
        )
        mp.chdir(tmp_path / "test_data")

        path = Path("poly.ert")
        text = path.read_text(encoding="utf-8")
        text = text.replace("NUM_REALIZATIONS 100", "NUM_REALIZATIONS 10")
        path.write_text(text, encoding="utf-8")

        poly_case = EnKFMain(ErtConfig.from_file("poly.ert"))
        args_mock = Mock()
        args_mock.config = "poly.ert"

        with StorageService.init_service(
            ert_config=args_mock.config,
            project=os.path.abspath(poly_case.ert_config.ens_path),
        ), open_storage(poly_case.ert_config.ens_path, mode="w") as storage:
            gui = _setup_main_window(poly_case, args_mock, GUILogHandler())
            gui.notifier.set_storage(storage)
            yield gui


@pytest.mark.usefixtures("use_tmpdir")
@pytest.fixture(scope="module")
def esmda_has_run(opened_main_window, request):
    qtbot = QtBot(request)
    gui, _ = opened_main_window
    qtbot.addWidget(gui)

    # Select Multiple Data Assimilation in the simulation panel
    simulation_panel = gui.findChild(SimulationPanel)
    assert isinstance(simulation_panel, SimulationPanel)
    simulation_mode_combo = simulation_panel.findChild(QComboBox)
    assert isinstance(simulation_mode_combo, QComboBox)
    current_select = 0
    simulation_mode_combo.setCurrentIndex(current_select)
    while simulation_mode_combo.currentText() != MultipleDataAssimilation.name():
        current_select += 1
        simulation_mode_combo.setCurrentIndex(current_select)

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

        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=120000)
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


@pytest.mark.usefixtures("use_tmpdir")
def test_that_the_plot_window_contains_the_expected_elements(
    esmda_has_run, opened_main_window, qtbot
):
    gui, _ = opened_main_window

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
        combo_box.setCurrentIndex(i)
        case_names.append(combo_box.currentText())
    assert sorted(case_names) == [
        "default_0",
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
    esmda_has_run,
    opened_main_window,
    qtbot,
):
    gui, _ = opened_main_window

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
def test_that_the_manual_analysis_tool_works(esmda_has_run, opened_main_window, qtbot):
    gui, _ = opened_main_window
    analysis_tool = gui.tools["Run analysis"]

    # Open the "Run analysis" tool in the main window after esmda has run
    def handle_analysis_dialog():
        dialog = analysis_tool._dialog

        # Set target case to "analysis_case"
        run_panel = analysis_tool._run_widget
        run_panel.target_case_text.setText("analysis_case")

        # Set source case to "default_3"
        case_selector = run_panel.source_case_selector
        index = case_selector.findText("default_3", Qt.MatchFlag.MatchContains)
        case_selector.setCurrentIndex(index)
        assert case_selector.currentText().startswith("default_3")

        # Click on "Run" and click ok on the message box
        def handle_dialog():
            messagebox = QApplication.activeModalWidget()
            assert isinstance(messagebox, QMessageBox)
            ok_button = messagebox.button(QMessageBox.Ok)
            qtbot.mouseClick(ok_button, Qt.LeftButton)

        QTimer.singleShot(1000, handle_dialog)
        qtbot.mouseClick(
            dialog.findChild(QPushButton, name="Run"),
            Qt.LeftButton,
        )

    QTimer.singleShot(1000, handle_analysis_dialog)
    analysis_tool.trigger()

    # Open the manage cases dialog
    def handle_manage_dialog():
        dialog = gui.findChild(ClosableDialog, name="manage_cases_tool")
        cases_panel = dialog.findChild(CaseInitializationConfigurationPanel)
        assert isinstance(cases_panel, CaseInitializationConfigurationPanel)

        # In the "create new case" tab, it should now contain "analysis_case"
        cases_panel.setCurrentIndex(0)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "create_new_case_tab"
        case_list = current_tab.findChild(CaseList)
        assert isinstance(case_list, CaseList)
        assert (
            len(case_list._list.findItems("analysis_case", Qt.MatchFlag.MatchContains))
            == 1
        )
        dialog.close()

    QTimer.singleShot(1000, handle_manage_dialog)
    manage_tool = gui.tools["Manage cases"]
    manage_tool.trigger()


@pytest.mark.usefixtures("use_tmpdir")
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
def test_that_the_load_results_manually_tool_works(
    esmda_has_run, opened_main_window, qtbot
):
    gui, ensemble_size = opened_main_window

    # Add a new case
    gui.notifier.storage.create_ensemble(
        gui.notifier.storage.create_experiment(),
        name="new_case_for_load",
        ensemble_size=ensemble_size,
    )
    gui.notifier.ertChanged.emit()

    def handle_load_results_dialog():
        qtbot.waitUntil(
            lambda: gui.findChild(ClosableDialog, name="load_results_manually_tool")
            is not None
        )
        dialog = gui.findChild(ClosableDialog, name="load_results_manually_tool")
        panel = dialog.findChild(LoadResultsPanel)
        assert isinstance(panel, LoadResultsPanel)

        combo_box = panel.findChild(QComboBox, "load_results_panel_case_combobox")
        assert isinstance(combo_box, QComboBox)
        index = combo_box.findText("new_case_for_load", Qt.MatchFlag.MatchContains)
        assert index != -1
        combo_box.setCurrentIndex(index)

        # click on "initialize"
        load_button = panel.parent().findChild(QPushButton, name="Load")
        assert isinstance(load_button, QPushButton)

        # Verify that the messagebox is the success kind
        def handle_popup_dialog():
            messagebox = QApplication.activeModalWidget()
            assert isinstance(messagebox, QMessageBox)
            assert messagebox.windowTitle() == "Success"
            ok_button = messagebox.button(QMessageBox.Ok)
            qtbot.mouseClick(ok_button, Qt.LeftButton)

        QTimer.singleShot(1000, handle_popup_dialog)
        qtbot.mouseClick(load_button, Qt.LeftButton)
        dialog.close()

    QTimer.singleShot(1000, handle_load_results_dialog)
    load_results_tool = gui.tools["Load results manually"]
    load_results_tool.trigger()
