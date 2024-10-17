import contextlib
import os
import shutil
import stat
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QAction,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QMenuBar,
    QPushButton,
    QToolButton,
    QTreeView,
    QWidget,
)
from xtgeo import RegularSurface

import ert.gui
from ert.config import ErtConfig
from ert.gui.about_dialog import AboutDialog
from ert.gui.ertwidgets import StringBox
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel
from ert.gui.ertwidgets.create_experiment_dialog import CreateExperimentDialog
from ert.gui.ertwidgets.ensembleselector import EnsembleSelector
from ert.gui.main import ErtMainWindow, GUILogHandler, _setup_main_window
from ert.gui.simulation.experiment_panel import ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.suggestor import Suggestor
from ert.gui.suggestor._suggestor_message import SuggestorMessage
from ert.gui.tools.event_viewer import add_gui_log_handler
from ert.gui.tools.manage_experiments import (
    ManageExperimentsPanel,
)
from ert.gui.tools.manage_experiments.storage_widget import AddWidget, StorageWidget
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_ensemble_selection_widget import (
    EnsembleSelectListWidget,
)
from ert.gui.tools.plot.plot_window import PlotApi, PlotWindow
from ert.plugins import ErtPluginManager
from ert.run_models import (
    EnsembleExperiment,
    EnsembleSmoother,
    IteratedEnsembleSmoother,
    MultipleDataAssimilation,
    SingleTestRun,
)
from ert.services import StorageService

from .conftest import (
    add_experiment_manually,
    get_child,
    load_results_manually,
    wait_for_child,
)


@pytest.mark.usefixtures("set_site_config")
@pytest.mark.parametrize(
    "config, expected_message_types",
    [
        (
            "NUM_REALIZATIONS 1\n"
            "INSTALL_JOB job job\n"
            "INSTALL_JOB job job\n"
            "FORWARD_MODEL not_installed\n",
            ["Error", "Warning"],
        ),
        ("NUM_REALIZATIONS you_cant_do_this\n", ["Error"]),
        ("NUM_REALIZATIONS 1\n UMASK 0222\n", ["Deprecation"]),
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
        assert isinstance(gui, Suggestor)
        suggestions = gui.findChildren(SuggestorMessage)
        shown_messages = [elem.lbl.text() for elem in suggestions]
        assert all(e in m for m, e in zip(shown_messages, expected_message_types))


@pytest.mark.usefixtures("copy_poly_case")
def test_that_the_ui_show_no_errors_and_enables_update_for_poly_example(qapp):
    args = Mock()
    args.config = "poly.ert"
    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(args, log_handler)
        combo_box = get_child(gui, QComboBox, name="experiment_type")
        assert combo_box.count() == 7

        for i in range(combo_box.count()):
            assert combo_box.model().item(i).isEnabled()

        assert gui.windowTitle().startswith("ERT - poly.ert")


@pytest.mark.usefixtures("set_site_config")
def test_gui_shows_a_warning_and_disables_update_when_there_are_no_observations(
    qapp, tmp_path
):
    config_file = tmp_path / "config.ert"
    config_file.write_text("NUM_REALIZATIONS 1\n")

    args = Mock()
    args.config = str(config_file)
    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(args, log_handler)
        combo_box = get_child(gui, QComboBox, name="experiment_type")
        assert combo_box.count() == 7

        for i in range(3):
            assert combo_box.model().item(i).isEnabled()
        for i in range(3, 5):
            assert not combo_box.model().item(i).isEnabled()

        assert gui.windowTitle().startswith("ERT - config.ert")


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
        combo_box = get_child(gui, QComboBox, name="experiment_type")
        assert combo_box.count() == 7

        for i in range(3):
            assert combo_box.model().item(i).isEnabled()
        for i in range(3, 5):
            assert not combo_box.model().item(i).isEnabled()

        assert gui.windowTitle().startswith("ERT - poly-no-gen-kw.ert")


@pytest.mark.usefixtures("set_site_config")
def test_help_buttons_in_suggester_dialog(tmp_path, qtbot):
    """
    WHEN I am shown an error in the gui
    THEN the suggester gui comes up
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
        assert isinstance(gui, Suggestor)

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
        assert gui.windowTitle().startswith("ERT - poly.ert")
        assert not gui.workflows_tool.getAction().isEnabled()


@pytest.mark.usefixtures("set_site_config")
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
        assert gui.windowTitle().startswith("ERT - config.ert")
        assert gui.workflows_tool.getAction().isEnabled()


@pytest.mark.usefixtures("copy_poly_case")
def test_that_es_mda_is_disabled_when_weights_are_invalid(qtbot):
    args = Mock()
    args.config = "poly.ert"
    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(args, log_handler)
        assert gui.windowTitle().startswith("ERT - poly.ert")

        combo_box = get_child(gui, QComboBox, name="experiment_type")
        combo_box.setCurrentIndex(3)

        assert combo_box.currentText() == "Multiple data assimilation"

        es_mda_panel = get_child(gui, QWidget, name="ES_MDA_panel")
        assert es_mda_panel

        run_sim_button = get_child(gui, QToolButton, name="run_experiment")
        assert run_sim_button
        assert run_sim_button.isEnabled()

        es_mda_panel._relative_iteration_weights_box.setText("0")

        assert not run_sim_button.isEnabled()

        es_mda_panel._relative_iteration_weights_box.setText("1")

        assert run_sim_button.isEnabled()


@pytest.mark.usefixtures("copy_snake_oil_field")
def test_that_ert_changes_to_config_directory(qtbot):
    """
    This is a regression test that verifies that ert changes directories
    to the config dir (where .ert is).
    Failure to do so would in this case result in SURFACE keyword not
    finding the INIT_FILE provided (surface/small.irap)
    """
    rng = np.random.default_rng()

    Path("./surface").mkdir()
    nx = 5
    ny = 10
    surf = RegularSurface(
        ncol=nx, nrow=ny, xinc=1.0, yinc=1.0, values=rng.standard_normal(size=(nx, ny))
    )
    surf.to_file("surface/surf_init_0.irap", fformat="irap_ascii")

    args = Mock()
    os.chdir("..")
    args.config = "test_data/snake_oil_surface.ert"
    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(args, log_handler)
        assert gui.windowTitle().startswith("ERT - snake_oil_surface.ert")


def test_that_the_plot_window_contains_the_expected_elements(
    esmda_has_run: ErtMainWindow, qtbot
):
    gui = esmda_has_run
    expected_ensembles = [
        "es_mda : default_0",
        "es_mda : default_1",
        "es_mda : default_2",
        "es_mda : default_3",
    ]

    # Click on Create plot after esmda has run
    button_plot_tool = gui.findChild(QToolButton, "button_Create_plot")
    assert button_plot_tool
    qtbot.mouseClick(button_plot_tool, Qt.LeftButton)
    plot_window = wait_for_child(gui, qtbot, PlotWindow)

    data_types = get_child(plot_window, DataTypeKeysWidget)
    case_selection = get_child(
        plot_window, EnsembleSelectListWidget, "ensemble_selector"
    )

    # Assert that the Case selection widget contains the expected ensembles
    ensemble_names = []
    for index in range(case_selection.count()):
        ensemble_names.append(case_selection.item(index).text())

    assert sorted(ensemble_names) == expected_ensembles

    data_names = []
    data_keys = data_types.data_type_keys_widget
    for i in range(data_keys.model().rowCount()):
        index = data_keys.model().index(i, 0)
        data_names.append(str(index.data(Qt.DisplayRole)))

    expected_data_names = [
        "POLY_RES@0",
        "COEFFS:a",
        "COEFFS:b",
        "COEFFS:c",
    ]
    expected_data_names.sort()
    data_names.sort()
    assert expected_data_names == data_names

    assert {
        plot_window._central_tab.tabText(i)
        for i in range(plot_window._central_tab.count())
    } == {
        "Cross ensemble statistics",
        "Distribution",
        "Gaussian KDE",
        "Ensemble",
        "Histogram",
        "Statistics",
        "Std Dev",
    }

    # Cycle through showing all the tabs and plot each data key

    model = data_keys.model()
    assert model is not None
    for i in range(model.rowCount()):
        index = model.index(i, 0)
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


def test_that_the_manage_experiments_tool_can_be_used(
    esmda_has_run,
    qtbot,
):
    gui = esmda_has_run

    button_manage_experiments = gui.findChild(QToolButton, "button_Manage_experiments")
    assert button_manage_experiments
    qtbot.mouseClick(button_manage_experiments, Qt.LeftButton)
    experiments_panel = wait_for_child(gui, qtbot, ManageExperimentsPanel)

    # Open the tab
    experiments_panel.setCurrentIndex(0)
    current_tab = experiments_panel.currentWidget()
    assert current_tab.objectName() == "create_new_ensemble_tab"

    storage_widget = get_child(current_tab, StorageWidget)
    tree_view = get_child(storage_widget, QTreeView)
    tree_view.expandAll()

    assert tree_view.model().rowCount() == 1
    assert tree_view.model().rowCount(tree_view.model().index(0, 0)) == 4

    def handle_add_dialog():
        dialog = wait_for_child(current_tab, qtbot, CreateExperimentDialog)

        dialog._experiment_edit.setText("es_mda")
        assert not dialog._ok_button.isEnabled()
        dialog._experiment_edit.setText(" @not_v alid")
        assert not dialog._ok_button.isEnabled()
        dialog._experiment_edit.setText("my-experiment")
        assert dialog._ok_button.isEnabled()

        dialog._ensemble_edit.setText(" @not_v alid")
        assert not dialog._ok_button.isEnabled()
        dialog._ensemble_edit.setText("_new_ensemble_")
        assert dialog._ok_button.isEnabled()

        qtbot.mouseClick(dialog._ok_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(1000, handle_add_dialog)
    create_widget = get_child(storage_widget, AddWidget)
    qtbot.mouseClick(create_widget.addButton, Qt.LeftButton)

    # Go to the "initialize from scratch" panel
    experiments_panel.setCurrentIndex(1)
    current_tab = experiments_panel.currentWidget()
    assert current_tab.objectName() == "initialize_from_scratch_panel"

    # click on "initialize"
    initialize_button = get_child(
        current_tab,
        QPushButton,
        name="initialize_from_scratch_button",
    )
    qtbot.mouseClick(initialize_button, Qt.LeftButton)


def test_that_inversion_type_can_be_set_from_gui(qtbot, opened_main_window_poly):
    gui = opened_main_window_poly

    sim_mode = get_child(gui, QWidget, name="experiment_type")
    qtbot.keyClick(sim_mode, Qt.Key_Down)
    es_panel = get_child(gui, QWidget, name="ensemble_smoother_panel")
    es_edit = get_child(es_panel, QWidget, name="ensemble_smoother_edit")

    # Testing modal dialogs requires some care.
    # https://github.com/pytest-dev/pytest-qt/issues/256
    def handle_analysis_module_panel():
        var_panel = wait_for_child(gui, qtbot, AnalysisModuleVariablesPanel)
        dropdown = wait_for_child(var_panel, qtbot, QComboBox)
        spinner = wait_for_child(var_panel, qtbot, QDoubleSpinBox, "enkf_truncation")
        assert [dropdown.itemText(i) for i in range(dropdown.count())] == [
            "EXACT",
            "SUBSPACE",
        ]
        for i in range(dropdown.count()):
            dropdown.setCurrentIndex(i)
            # spinner should be enabled if not rb0 set
            assert spinner.isEnabled() == (i != 0)

        var_panel.parent().close()

    QTimer.singleShot(500, handle_analysis_module_panel)
    qtbot.mouseClick(get_child(es_edit, QToolButton), Qt.LeftButton, delay=1)


def test_that_the_manage_experiments_tool_can_be_used_with_clean_storage(
    opened_main_window_poly, qtbot
):
    gui = opened_main_window_poly

    button_manage_experiments = gui.findChild(QToolButton, "button_Manage_experiments")
    assert button_manage_experiments
    qtbot.mouseClick(button_manage_experiments, Qt.LeftButton)
    experiments_panel = wait_for_child(gui, qtbot, ManageExperimentsPanel)

    # Open the create new ensembles tab
    experiments_panel.setCurrentIndex(0)
    current_tab = experiments_panel.currentWidget()
    assert current_tab.objectName() == "create_new_ensemble_tab"

    storage_widget = get_child(current_tab, StorageWidget)
    tree_view = get_child(storage_widget, QTreeView)
    tree_view.expandAll()

    assert tree_view.model().rowCount() == 0

    def handle_add_dialog():
        dialog = wait_for_child(current_tab, qtbot, CreateExperimentDialog)
        dialog._experiment_edit.setText("my-experiment")
        dialog._ensemble_edit.setText("_new_ensemble_")
        qtbot.mouseClick(dialog._ok_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(1000, handle_add_dialog)
    create_widget = get_child(storage_widget, AddWidget)
    qtbot.mouseClick(create_widget.addButton, Qt.MouseButton.LeftButton)

    assert tree_view.model().rowCount() == 1
    assert tree_view.model().rowCount(tree_view.model().index(0, 0)) == 1
    assert "_new_ensemble_" in tree_view.model().index(
        0, 0, tree_view.model().index(0, 0)
    ).data(0)

    # Go to the "initialize from scratch" panel
    experiments_panel.setCurrentIndex(1)
    current_tab = experiments_panel.currentWidget()
    assert current_tab.objectName() == "initialize_from_scratch_panel"
    combo_box = get_child(current_tab, EnsembleSelector)

    assert combo_box.currentText() == "my-experiment : _new_ensemble_"

    # click on "initialize"
    initialize_button = get_child(
        current_tab, QPushButton, name="initialize_from_scratch_button"
    )
    qtbot.mouseClick(initialize_button, Qt.LeftButton)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_load_results_manually_can_be_run_after_esmda(esmda_has_run, qtbot):
    load_results_manually(qtbot, esmda_has_run)


def test_that_a_failing_job_shows_error_message_with_context(
    opened_main_window_poly, qtbot
):
    gui = opened_main_window_poly

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
    experiment_panel = get_child(gui, ExperimentPanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(SingleTestRun.name())

    # Click start simulation and agree to the message
    run_experiment = get_child(experiment_panel, QWidget, name="run_experiment")

    def handle_error_dialog(run_dialog):
        qtbot.waitUntil(
            lambda: run_dialog.fail_msg_box is not None,
            timeout=20000,
        )
        error_dialog = run_dialog.fail_msg_box
        assert error_dialog
        text = error_dialog.details_text.toPlainText()
        label = error_dialog.label_text.text()
        assert "ERT experiment failed" in label
        expected_substrings = [
            "Realization: 0 failed after reaching max submit (1)",
            "job poly_eval failed",
            "Process exited with status code 1",
            "Traceback",
            "raise RuntimeError('Argh')",
            "RuntimeError: Argh",
        ]
        for substring in expected_substrings:
            assert substring in text
        error_dialog.accept()

    qtbot.mouseClick(run_experiment, Qt.LeftButton)

    run_dialog = wait_for_child(gui, qtbot, RunDialog)

    QTimer.singleShot(200, lambda: handle_error_dialog(run_dialog))
    qtbot.waitUntil(lambda: not run_dialog.done_button.isHidden(), timeout=100000)
    qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
def test_that_gui_plotter_works_when_no_data(qtbot, storage, monkeypatch):
    monkeypatch.setattr(PlotApi, "get_all_ensembles", lambda _: [])
    config_file = "minimal_config.ert"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1")
    args_mock = Mock()
    args_mock.config = config_file
    ert_config = ErtConfig.from_file(config_file)
    with StorageService.init_service(
        project=os.path.abspath(ert_config.ens_path),
    ):
        gui = _setup_main_window(ert_config, args_mock, GUILogHandler(), storage)
        qtbot.addWidget(gui)

        button_plot_tool = gui.findChild(QToolButton, "button_Create_plot")
        assert button_plot_tool
        qtbot.mouseClick(button_plot_tool, Qt.LeftButton)
        plot_window = wait_for_child(gui, qtbot, PlotWindow)

        ensemble_plot_names = get_child(
            plot_window, EnsembleSelectListWidget, "ensemble_selector"
        ).get_checked_ensembles()
        assert len(ensemble_plot_names) == 0


@pytest.mark.usefixtures("copy_poly_case")
def test_that_es_mda_restart_run_box_is_disabled_when_there_are_no_cases(qtbot):
    args = Mock()
    args.config = "poly.ert"
    ert_config = ErtConfig.from_file(args.config)
    with StorageService.init_service(
        project=os.path.abspath(ert_config.ens_path),
    ):
        gui, *_ = ert.gui.main._start_initial_gui_window(args, GUILogHandler())
        assert gui.windowTitle().startswith("ERT - poly.ert")

        combo_box = get_child(gui, QComboBox, name="experiment_type")
        qtbot.mouseClick(combo_box, Qt.MouseButton.LeftButton)
        assert combo_box.count() == 7
        combo_box.setCurrentIndex(3)

        assert combo_box.currentText() == "Multiple data assimilation"

        es_mda_panel = get_child(gui, QWidget, name="ES_MDA_panel")
        assert es_mda_panel

        restart_button = get_child(
            es_mda_panel, QCheckBox, name="restart_checkbox_esmda"
        )
        ensemble_selector = get_child(es_mda_panel, EnsembleSelector)

        assert restart_button

        assert len(ensemble_selector._ensemble_list()) == 0
        assert not restart_button.isEnabled()

        add_experiment_manually(qtbot, gui, ensemble_name="test_ensemble")
        assert len(ensemble_selector._ensemble_list()) == 1

        assert restart_button.isEnabled()


@pytest.mark.usefixtures("copy_poly_case")
def test_help_menu(qtbot):
    args = Mock()
    args.config = "poly.ert"
    ert_config = ErtConfig.from_file(args.config)
    with StorageService.init_service(
        project=os.path.abspath(ert_config.ens_path),
    ):
        gui, *_ = ert.gui.main._start_initial_gui_window(args, GUILogHandler())
        assert gui.windowTitle().startswith("ERT - poly.ert")
        menu_bar = gui.menuBar()
        assert isinstance(menu_bar, QMenuBar)
        get_child(menu_bar, QAction, name="about_action").trigger()
        about_dialog = wait_for_child(gui, qtbot, AboutDialog)
        assert about_dialog.windowTitle() == "About"
        qtbot.mouseClick(
            get_child(about_dialog, QPushButton, name="close_button"), Qt.LeftButton
        )


def test_validation_of_experiment_names_in_run_models(
    ensemble_experiment_has_run_no_failure, qtbot
):
    gui = ensemble_experiment_has_run_no_failure
    experiment_panel = get_child(gui, ExperimentPanel)
    experiment_types = get_child(experiment_panel, QComboBox, name="experiment_type")
    run_experiment = get_child(experiment_panel, QWidget, name="run_experiment")

    experiment_types_to_test = (
        (EnsembleExperiment.name(), "Ensemble_experiment_panel"),
        (EnsembleSmoother.name(), "ensemble_smoother_panel"),
        (MultipleDataAssimilation.name(), "ES_MDA_panel"),
        (IteratedEnsembleSmoother.name(), "iterated_ensemble_smoother_panel"),
    )
    for exp_type, panel_name in experiment_types_to_test:
        experiment_types.setCurrentText(exp_type)

        experiment_config_panel = get_child(gui, QWidget, name=panel_name)
        experiment_field = get_child(
            experiment_config_panel, StringBox, name="experiment_field"
        )
        experiment_field.setText(" @not val id")
        assert not run_experiment.isEnabled()

        experiment_field.setText("valid_")
        assert run_experiment.isEnabled()

        experiment_field.setText("ensemble_experiment")
        assert not run_experiment.isEnabled()
