import argparse
import contextlib
import os
import shutil
import stat
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QToolButton,
    QWidget,
)

import ert.gui
from ert.config import ErtConfig
from ert.enkf_main import EnKFMain
from ert.gui.about_dialog import AboutDialog
from ert.gui.ertwidgets import SuggestorMessage
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel
from ert.gui.ertwidgets.caselist import AddRemoveWidget, CaseList
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.customdialog import CustomDialog
from ert.gui.ertwidgets.listeditbox import ListEditBox
from ert.gui.ertwidgets.pathchooser import PathChooser
from ert.gui.ertwidgets.validateddialog import ValidatedDialog
from ert.gui.main import GUILogHandler, _setup_main_window, run_gui
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.simulation_panel import SimulationPanel
from ert.gui.tools.event_viewer import add_gui_log_handler
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_case_selection_widget import CaseSelectionWidget
from ert.gui.tools.plot.plot_window import PlotWindow
from ert.run_models import SingleTestRun
from ert.services import StorageService
from ert.shared.plugins.plugin_manager import ErtPluginManager
from tests.unit_tests.gui.simulation.test_run_path_dialog import handle_run_path_dialog

from .conftest import get_child, load_results_manually, wait_for_child, with_manage_tool


@pytest.mark.requires_window_manager
def test_that_loading_gui_creates_no_storage_in_read_only_mode(
    monkeypatch, tmp_path, qapp, source_root
):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(tmp_path, "poly_example"),
    )

    monkeypatch.chdir(tmp_path)

    args = argparse.Namespace(config="poly_example/poly.ert", read_only=True)

    qapp.exec_ = lambda: None  # exec_ starts the event loop, and will stall the test.
    monkeypatch.setattr(ert.gui.main, "QApplication", Mock(return_value=qapp))
    monkeypatch.setattr(ert.gui.main.LibresFacade, "enspath", tmp_path)
    run_gui(args)
    assert [p.stem for p in tmp_path.glob("**/*")].count("storage") == 0


@pytest.mark.usefixtures("use_tmpdir")
def test_gui_iter_num(monkeypatch, qtbot):
    config_file = Path("config.ert")
    config_file.write_text("NUM_REALIZATIONS 1\n", encoding="utf-8")

    args_mock = Mock()
    args_mock.config = str(config_file)

    # won't run simulations so we mock it and test whether "iter_num" is in arguments
    def _assert_iter_in_args(panel):
        assert panel.getSimulationArguments().iter_num == 10

    args_mock = Mock()
    args_mock.config = "poly.ert"
    type(args_mock).config = PropertyMock(return_value="config.ert")

    monkeypatch.setattr(
        ert.gui.simulation.simulation_panel.SimulationPanel,
        "runSimulation",
        _assert_iter_in_args,
    )

    gui = _setup_main_window(
        EnKFMain(ErtConfig.from_file(str(config_file))), args_mock, GUILogHandler()
    )
    qtbot.addWidget(gui)

    sim_mode = get_child(gui, QWidget, name="Simulation_mode")
    qtbot.keyClick(sim_mode, Qt.Key_Down)

    sim_panel = get_child(gui, QWidget, name="Simulation_panel")

    ensemble_panel = get_child(gui, QWidget, name="Ensemble_experiment_panel")
    # simulate entering number 10 as iter_num
    qtbot.keyClick(ensemble_panel._iter_field, Qt.Key_Backspace)
    qtbot.keyClicks(ensemble_panel._iter_field, "10")
    qtbot.keyClick(ensemble_panel._iter_field, Qt.Key_Enter)

    start_simulation = get_child(gui, QWidget, name="start_simulation")
    qtbot.mouseClick(start_simulation, Qt.LeftButton)
    assert sim_panel.getSimulationArguments().iter_num == 10


@pytest.mark.parametrize(
    "config, expected_message_types",
    [
        (
            "NUM_REALIZATIONS 1\n"
            "INSTALL_JOB job job\n"
            "INSTALL_JOB job job\n"
            "FORWARD_MODEL not_installed\n",
            ["ERROR", "WARNING"],
        ),
        ("NUM_REALIZATIONS you_cant_do_this\n", ["ERROR"]),
        ("NUM_REALIZATIONS 1\n UMASK 0222\n", ["DEPRECATION"]),
    ],
)
def test_both_errors_and_warning_can_be_shown_in_suggestor(
    qapp, tmp_path, config, expected_message_types
):
    config_file = tmp_path / "config.ert"
    job_file = tmp_path / "job"
    job_file.write_text("EXECUTABLE echo\n")
    config_file.write_text(config)

    args = Mock()
    args.config = str(config_file)
    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(args, log_handler)
        assert gui.windowTitle() == "Some problems detected"
        suggestions = gui.findChildren(SuggestorMessage)
        shown_messages = [elem.type_lbl.text() for elem in suggestions]
        assert shown_messages == expected_message_types


@pytest.mark.usefixtures("copy_poly_case")
def test_that_the_ui_show_no_errors_and_enables_update_for_poly_example(qapp):
    args = Mock()
    args.config = "poly.ert"
    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(args, log_handler)
        combo_box = get_child(gui, QComboBox, name="Simulation_mode")
        assert combo_box.count() == 5

        for i in range(combo_box.count()):
            assert combo_box.model().item(i).isEnabled()

        assert gui.windowTitle() == "ERT - poly.ert"


def test_gui_shows_a_warning_and_disables_update_when_there_are_no_observations(
    qapp, tmp_path
):
    config_file = tmp_path / "config.ert"
    config_file.write_text("NUM_REALIZATIONS 1\n")

    args = Mock()
    args.config = str(config_file)
    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(args, log_handler)
        combo_box = get_child(gui, QComboBox, name="Simulation_mode")
        assert combo_box.count() == 5

        for i in range(2):
            assert combo_box.model().item(i).isEnabled()
        for i in range(2, 5):
            assert not combo_box.model().item(i).isEnabled()

        assert gui.windowTitle() == "ERT - config.ert"


@pytest.mark.usefixtures("copy_poly_case")
def test_gui_shows_a_warning_and_disables_update_when_parameters_are_missing(
    qapp, tmp_path
):
    with open("poly.ert", "r", encoding="utf-8") as fin, open(
        "poly-no-gen-kw.ert", "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if "GEN_KW" not in line:
                fout.write(line)

    args = Mock()

    args.config = "poly-no-gen-kw.ert"
    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(args, log_handler)
        combo_box = get_child(gui, QComboBox, name="Simulation_mode")
        assert combo_box.count() == 5

        for i in range(2):
            assert combo_box.model().item(i).isEnabled()
        for i in range(2, 5):
            assert not combo_box.model().item(i).isEnabled()

        assert gui.windowTitle() == "ERT - poly-no-gen-kw.ert"


@pytest.mark.usefixtures("use_tmpdir")
def test_that_run_dialog_can_be_closed_after_used_to_open_plots(qtbot, storage):
    """
    This is a regression test for a bug where the plot window opened from run dialog
    would have run dialog as parent. Because of that it would be destroyed when
    run dialog was closed and end in a c++ QTObject lifetime crash.

    Also tests that the run_dialog is not modal (does not block the main_window),
    but simulations cannot be clicked from the main window while the run dialog is open.
    """
    config_file = Path("config.ert")
    config_file.write_text(
        f"NUM_REALIZATIONS 1\nENSPATH {storage.path}\n", encoding="utf-8"
    )

    args_mock = Mock()
    args_mock.config = str(config_file)

    ert_config = ErtConfig.from_file(str(config_file))
    enkf_main = EnKFMain(ert_config)
    with StorageService.init_service(
        ert_config=str(config_file),
        project=os.path.abspath(ert_config.ens_path),
    ):
        gui = _setup_main_window(enkf_main, args_mock, GUILogHandler())
        gui.notifier.set_storage(storage)
        qtbot.addWidget(gui)
        simulation_mode = get_child(gui, QComboBox, name="Simulation_mode")
        start_simulation = get_child(gui, QToolButton, name="start_simulation")

        def handle_dialog():
            message_box = wait_for_child(gui, qtbot, QMessageBox)
            qtbot.mouseClick(message_box.button(QMessageBox.Yes), Qt.LeftButton)

            QTimer.singleShot(
                500,
                lambda: handle_run_path_dialog(
                    gui=gui, qtbot=qtbot, delete_run_path=False
                ),
            )

        QTimer.singleShot(500, handle_dialog)
        qtbot.mouseClick(start_simulation, Qt.LeftButton)

        run_dialog = wait_for_child(gui, qtbot, RunDialog)

        # Ensure that once the run dialog is opened
        # another simulation cannot be started
        assert not start_simulation.isEnabled()

        # Change simulation mode and ensure that
        # another experiment still cannot be started
        for ind in range(simulation_mode.count()):
            simulation_mode.setCurrentIndex(ind)
            assert not start_simulation.isEnabled()

        # The user expects to be able to open e.g. the even viewer
        # while the run dialog is open
        assert not run_dialog.isModal()

        qtbot.mouseClick(run_dialog.plot_button, Qt.LeftButton)
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=20000)
        qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)

        # Ensure that once the run dialog is closed
        # another simulation can be started
        assert start_simulation.isEnabled()

        plot_window = wait_for_child(gui, qtbot, PlotWindow)

        # Cycle through showing all the tabs
        for tab in plot_window._plot_widgets:
            plot_window._central_tab.setCurrentWidget(tab)


def test_help_buttons_in_suggester_dialog(tmp_path, qtbot):
    """
    WHEN I am shown an error in the gui
    THEN the suggester gui comes up
    AND I can find the version of ert by opening the about panel
    AND go to github to submit an issue by clicking a button.
    """
    config_file = tmp_path / "config.ert"
    config_file.write_text(
        "NUM_REALIZATIONS 1\n RUNPATH iens-%d/iter-%d\n", encoding="utf-8"
    )

    args = Mock()
    args.config = str(config_file)
    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(
            args, log_handler, ErtPluginManager()
        )
        assert gui.windowTitle() == "Some problems detected"

        about_button = get_child(gui, QWidget, name="about_button")
        qtbot.mouseClick(about_button, Qt.LeftButton)

        help_dialog = get_child(gui, AboutDialog)
        assert help_dialog.windowTitle() == "About"

        about_close_button = get_child(help_dialog, QWidget, name="close_button")
        qtbot.mouseClick(about_close_button, Qt.LeftButton)

        with patch("webbrowser.open", MagicMock(return_value=True)) as browser_open:
            github_button = get_child(gui, QWidget, name="GitHub page")
            qtbot.mouseClick(github_button, Qt.LeftButton)
            assert browser_open.called


@pytest.mark.usefixtures("copy_poly_case")
def test_that_run_workflow_component_disabled_when_no_workflows(qapp):
    args = Mock()
    args.config = "poly.ert"
    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(args, log_handler)
        assert gui.windowTitle() == "ERT - poly.ert"
        run_workflow_button = gui.tools["Run workflow"]
        assert not run_workflow_button.isEnabled()


def test_that_run_workflow_component_enabled_when_workflows(qapp, tmp_path):
    config_file = tmp_path / "config.ert"

    with open(config_file, "a+", encoding="utf-8") as ert_file:
        ert_file.write("NUM_REALIZATIONS 1\n")
        ert_file.write("LOAD_WORKFLOW_JOB workflows/UBER_PRINT print_uber\n")
        ert_file.write("LOAD_WORKFLOW workflows/MAGIC_PRINT magic_print\n")

    os.mkdir(tmp_path / "workflows")

    with open(tmp_path / "workflows/MAGIC_PRINT", "w", encoding="utf-8") as f:
        f.write("print_uber\n")
    with open(tmp_path / "workflows/UBER_PRINT", "w", encoding="utf-8") as f:
        f.write("EXECUTABLE ls\n")

    args = Mock()
    args.config = str(config_file)

    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(args, log_handler)
        assert gui.windowTitle() == "ERT - config.ert"
        run_workflow_button = gui.tools["Run workflow"]
        assert run_workflow_button.isEnabled()


@pytest.mark.usefixtures("copy_poly_case")
def test_that_es_mda_is_disabled_when_weights_are_invalid(qtbot):
    args = Mock()
    args.config = "poly.ert"
    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(args, log_handler)
        assert gui.windowTitle() == "ERT - poly.ert"

        combo_box = get_child(gui, QComboBox, name="Simulation_mode")
        assert combo_box.count() == 5
        combo_box.setCurrentIndex(3)

        assert (
            combo_box.currentText()
            == "Multiple Data Assimilation (ES MDA) - Recommended"
        )

        es_mda_panel = get_child(gui, QWidget, name="ES_MDA_panel")
        assert es_mda_panel

        run_sim_button = get_child(gui, QToolButton, name="start_simulation")
        assert run_sim_button
        assert run_sim_button.isEnabled()

        es_mda_panel._relative_iteration_weights_box.setText("0")

        assert not run_sim_button.isEnabled()

        es_mda_panel._relative_iteration_weights_box.setText("1")

        assert run_sim_button.isEnabled()


@pytest.mark.usefixtures("copy_snake_oil_surface")
def test_that_ert_changes_to_config_directory(qtbot):
    """
    This is a regression test that verifies that ert changes directories
    to the config dir (where .ert is).
    Failure to do so would in this case result in SURFACE keyword not
    finding the INIT_FILE provided (surface/small.irap)
    """
    import numpy as np

    rng = np.random.default_rng()
    import xtgeo

    Path("./surface").mkdir()
    nx = 5
    ny = 10
    surf = xtgeo.RegularSurface(
        ncol=nx, nrow=ny, xinc=1.0, yinc=1.0, values=rng.standard_normal(size=(nx, ny))
    )
    surf.to_file("surface/surf_init_0.irap", fformat="irap_ascii")

    args = Mock()
    os.chdir("..")
    args.config = "test_data/snake_oil_surface.ert"
    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(args, log_handler)
        assert gui.windowTitle() == "ERT - snake_oil_surface.ert"


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
    plot_window = wait_for_child(gui, qtbot, PlotWindow)

    case_selection = get_child(plot_window, CaseSelectionWidget)
    data_types = get_child(plot_window, DataTypeKeysWidget)
    combo_box = get_child(case_selection, QComboBox, "case_selector")

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
        "COEFFS:a",
        "COEFFS:b",
        "COEFFS:c",
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
            get_child(case_selection, QToolButton, name="add_case_button"),
            Qt.LeftButton,
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
    def handle_dialog(dialog, cases_panel):
        # Open the create new cases tab
        cases_panel.setCurrentIndex(0)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "create_new_case_tab"
        create_widget = get_child(current_tab, AddRemoveWidget)
        case_list = get_child(current_tab, CaseList)

        # The case list should contain the expected cases
        assert case_list._list.count() == 5

        # Click add case and name it "new_case"
        def handle_add_dialog():
            dialog = wait_for_child(current_tab, qtbot, ValidatedDialog)
            dialog.param_name.setText("new_case")
            qtbot.mouseClick(dialog.ok_button, Qt.LeftButton)

        QTimer.singleShot(1000, handle_add_dialog)
        qtbot.mouseClick(create_widget.addButton, Qt.LeftButton)

        # The list should now contain "new_case"
        assert case_list._list.count() == 6

        # Click add case and try to name it "new_case" again
        def handle_add_dialog_again():
            dialog = wait_for_child(current_tab, qtbot, ValidatedDialog)
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
        combo_box = get_child(current_tab, CaseSelector)

        # Select "new_case"
        current_index = 0
        while combo_box.currentText().startswith("new_case"):
            current_index += 1
            combo_box.setCurrentIndex(current_index)

        # click on "initialize"
        initialize_button = get_child(
            current_tab,
            QPushButton,
            name="initialize_from_scratch_button",
        )
        qtbot.mouseClick(initialize_button, Qt.LeftButton)

        dialog.close()

    with_manage_tool(gui, qtbot, handle_dialog)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_inversion_type_can_be_set_from_gui(qtbot, opened_main_window):
    gui = opened_main_window

    sim_mode = get_child(gui, QWidget, name="Simulation_mode")
    qtbot.keyClick(sim_mode, Qt.Key_Down)
    es_panel = get_child(gui, QWidget, name="ensemble_smoother_panel")
    es_edit = get_child(es_panel, QWidget, name="ensemble_smoother_edit")

    # Testing modal dialogs requires some care.
    # https://github.com/pytest-dev/pytest-qt/issues/256
    def handle_analysis_module_panel():
        var_panel = wait_for_child(gui, qtbot, AnalysisModuleVariablesPanel)
        rb0 = wait_for_child(var_panel, qtbot, QRadioButton, name="IES_INVERSION_0")
        rb1 = wait_for_child(var_panel, qtbot, QRadioButton, name="IES_INVERSION_1")
        rb2 = wait_for_child(var_panel, qtbot, QRadioButton, name="IES_INVERSION_2")
        rb3 = wait_for_child(var_panel, qtbot, QRadioButton, name="IES_INVERSION_3")
        spinner = wait_for_child(var_panel, qtbot, QDoubleSpinBox, "ENKF_TRUNCATION")

        for b in [rb0, rb1, rb2, rb3, rb0]:
            b.click()
            assert b.isChecked()
            # spinner should be enabled if not rb0 set
            assert spinner.isEnabled() == (b != rb0)

        var_panel.parent().close()

    QTimer.singleShot(500, handle_analysis_module_panel)
    qtbot.mouseClick(get_child(es_edit, QToolButton), Qt.LeftButton, delay=1)


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
        export_dialog = wait_for_child(gui, qtbot, CustomDialog)
        case_selection = get_child(export_dialog, ListEditBox)

        # Select default_0 as the case to be exported
        case_selection._list_edit_line.setText("default_0")
        path_chooser = get_child(export_dialog, PathChooser)
        file_name = path_chooser._path_line.text()
        assert case_selection.isValid()

        qtbot.mouseClick(export_dialog.ok_button, Qt.LeftButton)

    def handle_finished_box():
        """
        Click on the plugin finised dialog once it pops up
        """
        finished_message = wait_for_child(gui, qtbot, QMessageBox)
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
    def handle_dialog(dialog, cases_panel):
        # Open the create new cases tab
        cases_panel.setCurrentIndex(0)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "create_new_case_tab"
        create_widget = get_child(current_tab, AddRemoveWidget)
        case_list = get_child(current_tab, CaseList)

        assert case_list._list.count() == 0

        # Click add case and name it "new_case"
        def handle_add_dialog():
            dialog = wait_for_child(current_tab, qtbot, ValidatedDialog)
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
        combo_box = get_child(current_tab, CaseSelector)

        assert combo_box.currentText().startswith("new_case")

        # click on "initialize"
        initialize_button = get_child(
            current_tab, QPushButton, name="initialize_from_scratch_button"
        )
        qtbot.mouseClick(initialize_button, Qt.LeftButton)

        dialog.close()

    with_manage_tool(gui, qtbot, handle_dialog)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_load_results_manually_can_be_run_after_esmda(
    esmda_has_run, opened_main_window, qtbot
):
    load_results_manually(qtbot, opened_main_window)


@pytest.mark.skip(reason="Needs reimplementation")
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
    simulation_panel = get_child(gui, SimulationPanel)
    simulation_mode_combo = get_child(simulation_panel, QComboBox)
    simulation_mode_combo.setCurrentText(SingleTestRun.name())

    # Click start simulation and agree to the message
    start_simulation = get_child(simulation_panel, QWidget, name="start_simulation")

    def handle_dialog():
        qtbot.mouseClick(
            wait_for_child(gui, qtbot, QMessageBox).buttons()[0], Qt.LeftButton
        )

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

    run_dialog = wait_for_child(gui, qtbot, RunDialog)
    qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)

    QTimer.singleShot(20000, lambda: handle_error_dialog(run_dialog))
    qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_gui_plotter_disables_add_case_button_when_no_data(qtbot, storage):
    config_file = "minimal_config.ert"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1")
    args_mock = Mock()
    args_mock.config = config_file

    ert_config = ErtConfig.from_file(config_file)
    enkf_main = EnKFMain(ert_config)
    with StorageService.init_service(
        ert_config=config_file,
        project=os.path.abspath(ert_config.ens_path),
    ):
        gui = _setup_main_window(enkf_main, args_mock, GUILogHandler())
        gui.notifier.set_storage(storage)
        qtbot.addWidget(gui)
        gui.tools["Create plot"].trigger()

        wait_for_child(gui, qtbot, PlotWindow)

        add_case_button = gui.findChild(QToolButton, name="add_case_button")
        assert add_case_button
        assert not add_case_button.isEnabled()
