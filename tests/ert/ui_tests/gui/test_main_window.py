import contextlib
import logging
import os
import shutil
import stat
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from ert.tests.handle_run_path_dialog import handle_run_path_dialog
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QToolButton,
    QTreeView,
    QWidget,
)
from xtgeo import RegularSurface

import ert.gui
from ert.config import ErtConfig
from ert.gui.about_dialog import AboutDialog
from ert.gui.ertwidgets import (
    CreateExperimentDialog,
    EnsembleSelector,
    StringBox,
    Suggestor,
)
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel
from ert.gui.ertwidgets.suggestor._suggestor_message import SuggestorMessage
from ert.gui.experiments import ExperimentPanel, RunDialog
from ert.gui.main import ErtMainWindow, GUILogHandler, _setup_main_window
from ert.gui.main_window import SidebarToolButton
from ert.gui.tools.event_viewer import add_gui_log_handler
from ert.gui.tools.manage_experiments import ManageExperimentsPanel
from ert.gui.tools.manage_experiments.storage_widget import AddWidget, StorageWidget
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_ensemble_selection_widget import EnsembleSelectListWidget
from ert.gui.tools.plot.plot_window import (
    GEN_KW_DEFAULT,
    RESPONSE_DEFAULT,
    PlotApi,
    PlotWindow,
)
from ert.plugins import get_site_plugins
from ert.run_models import (
    EnsembleExperiment,
    EnsembleInformationFilter,
    EnsembleSmoother,
    MultipleDataAssimilation,
    SingleTestRun,
)
from ert.services import ErtServerController
from ert.storage import open_storage

from .conftest import (
    add_experiment_manually,
    get_child,
    get_children,
    load_results_manually,
    wait_for_child,
)


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@pytest.mark.parametrize(
    ("config", "expected_message_types"),
    [
        (
            (
                "NUM_REALIZATIONS 1\n"
                "INSTALL_JOB job job\n"
                "INSTALL_JOB job job\n"
                "FORWARD_MODEL not_installed\n"
            ),
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
        assert all(
            e in m for m, e in zip(shown_messages, expected_message_types, strict=False)
        )


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


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
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
def test_gui_shows_a_warning_and_disables_update_when_parameters_are_missing(qapp):
    with (
        open("poly.ert", encoding="utf-8") as fin,
        open("poly-no-gen-kw.ert", "w", encoding="utf-8") as fout,
    ):
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


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_there_is_a_link_to_github_in_the_suggester(tmp_path, qtbot):
    """
    WHEN I am shown an error in the gui
    THEN the suggester gui comes up
    AND I can go to github to submit an issue by clicking a button.
    """
    config_file = tmp_path / "config.ert"
    config_file.write_text(
        "NUM_REALIZATIONS 1\n RUNPATH iens-%d/iter-%d\n", encoding="utf-8"
    )

    args = Mock()
    args.config = str(config_file)
    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(
            args, log_handler, get_site_plugins()
        )
        assert isinstance(gui, Suggestor)

        with patch("webbrowser.open", MagicMock(return_value=True)) as browser_open:
            github_button = get_child(gui, QWidget, name="GitHub page")
            qtbot.mouseClick(github_button, Qt.MouseButton.LeftButton)
            assert browser_open.called


@pytest.mark.usefixtures("copy_poly_case")
def test_that_the_run_workflow_tool_is_disabled_when_there_are_no_workflows(qapp):
    args = Mock()
    args.config = "poly.ert"
    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(args, log_handler)
        assert gui.windowTitle().startswith("ERT - poly.ert")
        assert not gui.workflows_tool.getAction().isEnabled()


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_the_run_workflow_tool_is_enabled_when_there_are_workflows(qapp, tmp_path):
    config_file = tmp_path / "config.ert"

    with open(config_file, "a+", encoding="utf-8") as ert_file:
        ert_file.write("NUM_REALIZATIONS 1\n")
        ert_file.write("LOAD_WORKFLOW_JOB workflows/UBER_PRINT print_uber\n")
        ert_file.write("LOAD_WORKFLOW workflows/MAGIC_PRINT magic_print\n")

    os.mkdir(tmp_path / "workflows")

    Path(tmp_path / "workflows/MAGIC_PRINT").write_text(
        "print_uber\n", encoding="utf-8"
    )
    Path(tmp_path / "workflows/UBER_PRINT").write_text(
        "EXECUTABLE ls\n", encoding="utf-8"
    )

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

        assert combo_box.currentText() == MultipleDataAssimilation.display_name()

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


@pytest.mark.skip_mac_ci
def test_that_the_plot_window_contains_the_expected_elements(
    esmda_has_run: ErtMainWindow, qtbot
):
    gui = esmda_has_run
    open_storage(gui.ert_config.ens_path, mode="r")
    with ErtServerController.init_service(
        project=os.path.abspath(gui.ert_config.ens_path),
    ):
        expected_ensembles = [
            "es_mda : default_0",
            "es_mda : default_1",
            "es_mda : default_2",
            "es_mda : default_3",
        ]

        # Click on Create plot after esmda has run
        button_plot_tool = gui.findChild(QToolButton, "button_Create_plot")
        assert button_plot_tool
        qtbot.mouseClick(button_plot_tool, Qt.MouseButton.LeftButton)
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
            data_names.append(str(index.data(Qt.ItemDataRole.DisplayRole)))

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
            "Misfits",
        }

        model = data_keys.model()
        assert model is not None

        def tab_index_by_text(text: str) -> int:
            for i in range(plot_window._central_tab.count()):
                if plot_window._central_tab.tabText(i) == text:
                    return i
            raise AssertionError(f"Tab '{text}' not found")

        def click_plotter_item(pos: int) -> None:
            center = data_keys.visualRect(model.index(pos, 0)).center()
            viewport = data_keys.viewport()
            center = viewport.mapToGlobal(center)
            local_pos = viewport.mapFromGlobal(center)
            qtbot.mouseClick(
                data_keys.viewport(), Qt.MouseButton.LeftButton, pos=local_pos
            )

        def click_tab_index(pos: int) -> None:
            tab_bar = plot_window._central_tab.tabBar()
            tab_center = tab_bar.tabRect(pos).center()
            qtbot.mouseClick(tab_bar, Qt.MouseButton.LeftButton, pos=tab_center)

        def get_log_checkbox():
            w = plot_window._central_tab.currentWidget()
            return w.findChild(QCheckBox, "log_scale_checkbox")

        # make sure plotter remembers plot types selected previously
        response_index = 0  # responses are at the start, thus POLY_RES@0 is at index0
        gen_kw_index = 3
        response_alternate_index = 1
        gen_kw_alternate_index = tab_index_by_text("Histogram")

        # check default selections
        click_plotter_item(response_index)
        assert plot_window._central_tab.currentIndex() == RESPONSE_DEFAULT

        # no log scale checkbox yet
        cb = get_log_checkbox()
        assert not cb.isVisibleTo(plot_window._central_tab.currentWidget())

        click_plotter_item(gen_kw_index)
        assert plot_window._central_tab.currentIndex() == GEN_KW_DEFAULT

        # alter selections
        click_plotter_item(response_index)
        click_tab_index(response_alternate_index)
        click_plotter_item(gen_kw_index)
        click_tab_index(gen_kw_alternate_index)

        # verify previous selections still valid
        click_plotter_item(response_index)
        assert plot_window._central_tab.currentIndex() == response_alternate_index
        click_plotter_item(gen_kw_index)
        assert plot_window._central_tab.currentIndex() == gen_kw_alternate_index

        # wait until the checkbox exists
        qtbot.waitUntil(lambda: get_log_checkbox() is not None, timeout=2000)
        cb = get_log_checkbox()

        # wait until it becomes visible
        qtbot.waitUntil(
            lambda: cb.isVisibleTo(plot_window._central_tab.currentWidget()) is True,
            timeout=2000,
        )

        # finally click all items
        for i in range(model.rowCount()):
            click_plotter_item(i)
            for tab_index in range(plot_window._central_tab.count()):
                if not plot_window._central_tab.isTabEnabled(tab_index):
                    continue
                click_tab_index(tab_index)

        plot_window.close()


def test_that_the_manage_experiments_tool_can_be_used(esmda_has_run, qtbot):
    gui = esmda_has_run
    button_manage_experiments = gui.findChild(QToolButton, "button_Manage_experiments")
    assert button_manage_experiments
    qtbot.mouseClick(button_manage_experiments, Qt.MouseButton.LeftButton)
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

        dialog._iterations_field.setText("a")
        assert not dialog._ok_button.isEnabled()
        dialog._iterations_field.setText("42")
        assert dialog._ok_button.isEnabled()

        qtbot.mouseClick(dialog._ok_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(1000, handle_add_dialog)
    create_widget = get_child(storage_widget, AddWidget)
    qtbot.mouseClick(create_widget.addButton, Qt.MouseButton.LeftButton)

    assert experiments_panel.notifier.current_ensemble.iteration == 42

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
    qtbot.mouseClick(initialize_button, Qt.MouseButton.LeftButton)


def test_that_inversion_type_can_be_set_from_gui(qtbot, opened_main_window_poly):
    gui = opened_main_window_poly

    sim_mode = get_child(gui, QWidget, name="experiment_type")
    qtbot.keyClick(sim_mode, Qt.Key.Key_Down)
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
    qtbot.mouseClick(
        get_child(es_edit, QToolButton), Qt.MouseButton.LeftButton, delay=1
    )


def test_that_the_manage_experiments_tool_can_be_used_with_clean_storage(
    opened_main_window_poly, qtbot
):
    gui = opened_main_window_poly

    button_manage_experiments = gui.findChild(QToolButton, "button_Manage_experiments")
    assert button_manage_experiments
    qtbot.mouseClick(button_manage_experiments, Qt.MouseButton.LeftButton)
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
    qtbot.mouseClick(initialize_button, Qt.MouseButton.LeftButton)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_load_results_manually_can_be_run_after_esmda(esmda_has_run, qtbot):
    load_results_manually(qtbot, esmda_has_run)


def test_that_a_failing_job_shows_error_message_with_context(
    opened_main_window_poly, qtbot, use_tmpdir
):
    gui = opened_main_window_poly

    # break poly eval script so realz fail
    Path("poly_eval.py").write_text(
        dedent(
            """\
                #!/usr/bin/env python

                if __name__ == "__main__":
                    raise RuntimeError('Argh')
                """
        ),
        encoding="utf-8",
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
        error_dialog: Suggestor = run_dialog.fail_msg_box
        assert error_dialog

        assert "ERT experiment failed" in error_dialog.findChild(QLabel).text()
        expected_substrings = [
            "Realization: 0 failed after reaching max submit (1)",
            "Step poly_eval failed",
            "Process exited with status code 1",
            "Traceback",
            "raise RuntimeError('Argh')",
            "RuntimeError: Argh",
        ]
        suggestor_messages = (
            error_dialog.findChild(QWidget, name="suggestor_messages")
            .findChild(QLabel)
            .text()
        )
        for substring in expected_substrings:
            assert substring in suggestor_messages
        error_dialog.close()

    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)

    run_dialog = wait_for_child(gui, qtbot, RunDialog)

    QTimer.singleShot(200, lambda: handle_error_dialog(run_dialog))
    qtbot.waitUntil(lambda: run_dialog.is_experiment_done() is True, timeout=100000)


@pytest.mark.skip_mac_ci
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_gui_plotter_works_when_no_data(qtbot, monkeypatch, use_tmpdir):
    monkeypatch.setattr(PlotApi, "get_all_ensembles", lambda _: [])
    config_file = "minimal_config.ert"
    Path(config_file).write_text(
        "NUM_REALIZATIONS 1\nENSPATH storage\nQUEUE_SYSTEM LOCAL", encoding="utf-8"
    )

    args_mock = Mock()
    args_mock.config = config_file
    ert_config = ErtConfig.from_file(config_file)
    # Open up storage to create it, so that dark storage can be mounted onto it
    # Not creating will result in dark storage hanging/lagging
    open_storage(ert_config.ens_path, mode="r")

    with ErtServerController.init_service(
        project=os.path.abspath(ert_config.ens_path),
    ):
        gui = _setup_main_window(
            ert_config, args_mock, GUILogHandler(), ert_config.ens_path
        )
        qtbot.addWidget(gui)

        button_plot_tool = gui.findChild(QToolButton, "button_Create_plot")
        assert button_plot_tool
        qtbot.mouseClick(button_plot_tool, Qt.MouseButton.LeftButton)
        plot_window = wait_for_child(gui, qtbot, PlotWindow)

        ensemble_plot_names = get_child(
            plot_window, EnsembleSelectListWidget, "ensemble_selector"
        ).get_checked_ensembles()
        assert len(ensemble_plot_names) == 0


@pytest.mark.skip_mac_ci
@pytest.mark.usefixtures("use_tmpdir", "use_site_configurations_with_no_queue_options")
def test_right_click_plot_button_opens_external_plotter(qtbot, use_tmpdir, monkeypatch):
    monkeypatch.setattr(PlotApi, "get_all_ensembles", lambda _: [])
    config_file = "minimal_config.ert"
    Path(config_file).write_text(
        "NUM_REALIZATIONS 1\nENSPATH storage\nQUEUE_SYSTEM LOCAL", encoding="utf-8"
    )

    # Open up storage to create it, so that dark storage can be mounted onto it
    # Not creating will result in dark storage hanging/lagging
    open_storage("storage", mode="r")

    args_mock = Mock()
    args_mock.config = config_file
    ert_config = ErtConfig.from_file(config_file)
    with ErtServerController.init_service(
        project=os.path.abspath(ert_config.ens_path),
    ):
        gui = _setup_main_window(
            ert_config, args_mock, GUILogHandler(), ert_config.ens_path
        )
        qtbot.addWidget(gui)

        button_plot_tool = gui.findChild(QToolButton, "button_Create_plot")
        assert button_plot_tool

        def top_level_plotter_windows() -> list[PlotWindow]:
            plot_windows = gui.get_external_plot_windows()
            top_level_plot_windows = []

            for win in plot_windows:
                if "Plotting" in win.windowTitle() and win.isVisible():
                    top_level_plot_windows.append(win)
            return top_level_plot_windows

        def right_click_plotter_button() -> None:
            top_level_windows = len(top_level_plotter_windows())
            qtbot.mouseClick(button_plot_tool, Qt.MouseButton.RightButton)
            qtbot.wait_until(
                lambda: len(top_level_plotter_windows()) > top_level_windows,
                timeout=5000,
            )

        right_click_plotter_button()
        right_click_plotter_button()
        right_click_plotter_button()

        window_list = top_level_plotter_windows()
        assert len(window_list) == 3

        for window in window_list:
            window.close()

        qtbot.wait_until(lambda: not top_level_plotter_windows(), timeout=5000)

        qtbot.mouseClick(button_plot_tool, Qt.MouseButton.LeftButton)
        plot_window = wait_for_child(gui, qtbot, PlotWindow)
        assert plot_window
        assert "Plotting" in plot_window.windowTitle()

    gui.close()


@pytest.mark.usefixtures("copy_poly_case")
def test_that_es_mda_restart_run_box_is_disabled_when_there_are_no_cases(qtbot):
    args = Mock()
    args.config = "poly.ert"
    gui, *_ = ert.gui.main._start_initial_gui_window(args, GUILogHandler())
    assert gui.windowTitle().startswith("ERT - poly.ert")

    combo_box = get_child(gui, QComboBox, name="experiment_type")
    qtbot.mouseClick(combo_box, Qt.MouseButton.LeftButton)
    assert combo_box.count() == 7
    combo_box.setCurrentIndex(3)

    assert combo_box.currentText() == MultipleDataAssimilation.display_name()

    es_mda_panel = get_child(gui, QWidget, name="ES_MDA_panel")
    assert es_mda_panel

    restart_button = get_child(es_mda_panel, QCheckBox, name="restart_checkbox_esmda")
    ensemble_selector = get_child(es_mda_panel, EnsembleSelector)

    assert restart_button

    assert len(ensemble_selector._ensemble_list()) == 0
    assert not restart_button.isEnabled()

    add_experiment_manually(qtbot, gui, ensemble_name="test_ensemble")
    assert len(ensemble_selector._ensemble_list()) == 1

    assert restart_button.isEnabled()


@pytest.mark.usefixtures("copy_poly_case")
def test_that_the_help_menu_contains_the_about_dialog(qtbot):
    args = Mock()
    args.config = "poly.ert"
    gui, *_ = ert.gui.main._start_initial_gui_window(args, GUILogHandler())
    assert gui.windowTitle().startswith("ERT - poly.ert")
    menu_bar = gui.menuBar()
    assert isinstance(menu_bar, QMenuBar)
    get_child(menu_bar, QAction, name="about_action").trigger()
    about_dialog = wait_for_child(gui, qtbot, AboutDialog)
    assert about_dialog.windowTitle() == "About"
    qtbot.mouseClick(
        get_child(about_dialog, QPushButton, name="close_button"),
        Qt.MouseButton.LeftButton,
    )


@pytest.mark.parametrize(
    ("exp_type", "panel_name"),
    [
        (EnsembleExperiment, "Ensemble_experiment_panel"),
        (EnsembleSmoother, "ensemble_smoother_panel"),
        (EnsembleInformationFilter, "enif_panel"),
        (MultipleDataAssimilation, "ES_MDA_panel"),
    ],
)
def test_that_the_run_experiment_button_is_disabled_when_the_experiment_name_is_invalid(
    opened_main_window_poly, qtbot, exp_type, panel_name
):
    gui = opened_main_window_poly
    experiment_panel = get_child(gui, ExperimentPanel)
    experiment_types = get_child(experiment_panel, QComboBox, name="experiment_type")
    run_experiment = get_child(experiment_panel, QWidget, name="run_experiment")

    experiment_types.setCurrentText(exp_type.display_name())

    experiment_config_panel = get_child(gui, QWidget, name=panel_name)
    experiment_field = get_child(
        experiment_config_panel, StringBox, name="experiment_field"
    )
    experiment_field.setText(" @not val id")
    assert not run_experiment.isEnabled()

    experiment_field.setText("valid_")
    assert run_experiment.isEnabled()


def test_that_simulation_status_button_adds_menu_on_subsequent_runs(
    opened_main_window_poly, qtbot
):
    gui = opened_main_window_poly

    def find_and_click_button(
        button_name: str, should_click: bool, expected_enabled_state: bool
    ):
        button = gui.findChild(SidebarToolButton, button_name)
        assert button
        assert button.isEnabled() == expected_enabled_state
        if should_click:
            qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    def find_and_check_selected(button_name: str, expected_selected_state: bool):
        button = gui.findChild(SidebarToolButton, button_name)
        assert button
        assert button.isChecked() == expected_selected_state

    def run_experiment():
        run_experiment_panel = wait_for_child(gui, qtbot, ExperimentPanel)
        qtbot.wait_until(lambda: not run_experiment_panel.isHidden(), timeout=5000)
        assert run_experiment_panel.run_button.isEnabled()
        qtbot.mouseClick(run_experiment_panel.run_button, Qt.MouseButton.LeftButton)

    def wait_for_simulation_completed():
        run_dialogs = get_children(gui, RunDialog)
        dialog = run_dialogs[-1]
        qtbot.wait_until(lambda: not dialog.isHidden(), timeout=5000)
        qtbot.wait_until(lambda: dialog.is_experiment_done() is True, timeout=15000)

    # not clickable since no simulations started yet
    find_and_click_button("button_Simulation_status", False, False)
    find_and_click_button("button_Start_simulation", True, True)

    run_experiment()
    wait_for_simulation_completed()

    # just toggle to see if next button yields intended change
    find_and_click_button("button_Start_simulation", True, True)
    experiments_panel = wait_for_child(gui, qtbot, ExperimentPanel)
    qtbot.wait_until(lambda: not experiments_panel.isHidden(), timeout=5000)

    find_and_click_button("button_Simulation_status", True, True)
    run_dialog = wait_for_child(gui, qtbot, RunDialog)
    qtbot.wait_until(lambda: not run_dialog.isHidden(), timeout=5000)

    # verify no drop menu
    button_simulation_status = gui.findChild(
        SidebarToolButton, "button_Simulation_status"
    )
    assert button_simulation_status.menu() is None

    find_and_click_button("button_Start_simulation", True, True)
    QTimer.singleShot(500, lambda: handle_run_path_dialog(gui, qtbot, True))
    run_experiment()
    wait_for_simulation_completed()

    # verify menu available
    assert len(button_simulation_status.menu().actions()) == 2

    find_and_click_button("button_Start_simulation", True, True)
    QTimer.singleShot(500, lambda: handle_run_path_dialog(gui, qtbot, True))
    run_experiment()
    wait_for_simulation_completed()

    # click on something else just to shift focus
    find_and_click_button("button_Start_simulation", True, True)
    # verify correct button in focus
    find_and_check_selected("button_Start_simulation", True)
    find_and_check_selected("button_Simulation_status", False)

    assert len(button_simulation_status.menu().actions()) == 3
    for choice in button_simulation_status.menu().actions():
        assert "Single realization test-run" in choice.text()

        # verify correct button in focus when selecting from drop-down
        choice.trigger()
        find_and_check_selected("button_Start_simulation", False)
        find_and_check_selected("button_Simulation_status", True)


def test_that_visible_experiment_label_matches_bold_simulation_menu_action(
    opened_main_window_poly, qtbot, run_experiment
):
    gui = opened_main_window_poly

    runs = {}
    actions = {}
    titles = []

    def retrieve_latest_dialog_title(expected_dialogs_number):
        run_dialogs = get_children(gui, RunDialog)
        assert len(run_dialogs) == expected_dialogs_number
        latest_run_dialog = run_dialogs[-1]
        latest_run_title = latest_run_dialog._experiment_name_label.text()
        # all titles should be different because run_experiment fixture takes >1s
        assert latest_run_title not in titles
        titles.append(latest_run_title)
        runs[latest_run_title] = latest_run_dialog

    def retrieve_existing_menu_actions(expected_actions_number):
        simulation_status_menu = gui.findChild(
            SidebarToolButton, "button_Simulation_status"
        ).menu()
        assert len(simulation_status_menu.actions()) == expected_actions_number
        for action in simulation_status_menu.actions():
            actions[action.text()] = action

    def verify_only_latest_run_is_marked_bold_and_dialog_visible_on_new_experiment():
        for title in titles[:-1]:
            assert not actions[title].font().bold()
            assert runs[title].isHidden()

        latest_run_title = titles[-1]
        assert actions[latest_run_title].font().bold()
        assert not runs[latest_run_title].isHidden()

    def verify_clicking_on_action_changes_bold_action_and_visible_dialog():
        first_run_title = titles[0]
        first_run_dialog = runs[first_run_title]

        latest_run_title = titles[-1]
        currently_active_dialog = runs[latest_run_title]

        first_run_action = actions[first_run_title]
        first_run_action.trigger()
        qtbot.wait_until(currently_active_dialog.isHidden, timeout=5000)
        assert not first_run_dialog.isHidden()
        assert actions[first_run_title].font().bold()

    run_experiment(SingleTestRun, gui)
    retrieve_latest_dialog_title(1)

    run_experiment(SingleTestRun, gui)
    retrieve_latest_dialog_title(2)
    retrieve_existing_menu_actions(2)
    verify_only_latest_run_is_marked_bold_and_dialog_visible_on_new_experiment()
    verify_clicking_on_action_changes_bold_action_and_visible_dialog()

    run_experiment(SingleTestRun, gui)
    retrieve_latest_dialog_title(3)
    retrieve_existing_menu_actions(3)
    verify_only_latest_run_is_marked_bold_and_dialog_visible_on_new_experiment()
    verify_clicking_on_action_changes_bold_action_and_visible_dialog()


def test_warnings_from_forward_model_are_propagated_to_ert_main_window_post_simulation(
    qtbot, opened_main_window_poly, use_tmpdir
):
    forward_model_file = Path("WARNING_EXAMPLE")
    forward_model_file.write_text(
        """
    EXECUTABLE warning.py""",
        encoding="utf-8",
    )

    script_file = "warning.py"
    script_file_content = """#!/usr/bin/env python
import warnings
warnings.warn('Foobar')"""

    Path(script_file).write_text(dedent(script_file_content), encoding="utf-8")

    os.chmod(
        script_file,
        os.stat(script_file).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )

    config_file = "config.ert"
    config_file_content = """\
            NUM_REALIZATIONS 1
            INSTALL_JOB poly_eval WARNING_EXAMPLE
            FORWARD_MODEL poly_eval
        """
    Path(config_file).write_text(dedent(config_file_content), encoding="utf-8")

    ert_config = ErtConfig.from_file(config_file)
    args_mock = Mock()
    args_mock.config = config_file

    gui = _setup_main_window(
        ert_config, args_mock, GUILogHandler(), ert_config.ens_path
    )
    qtbot.addWidget(gui)

    experiments_panel = wait_for_child(gui, qtbot, ExperimentPanel)
    qtbot.wait_until(lambda: not experiments_panel.isHidden(), timeout=5000)
    qtbot.mouseClick(experiments_panel.run_button, Qt.MouseButton.LeftButton)

    run_dialog = wait_for_child(gui, qtbot, RunDialog)

    qtbot.wait_until(lambda: run_dialog.fail_msg_box is not None, timeout=20000)
    expected_messages = [
        "ERT experiment succeeded!",
        "These warnings were detected",
        "UserWarning: Foobar",
    ]
    messages = "\n".join(
        [child.text() for child in run_dialog.fail_msg_box.findChildren(QLabel)]
    )

    for expected_message in expected_messages:
        assert expected_message in messages

    # Regression test for total progress bar being green given
    # PostExperimentWarning and no failures
    assert (
        run_dialog._total_progress_label.text()
        == "Total progress 100% — Experiment completed."
    )

    # Test that show_warnings_button enables and disables
    # the suggestor window containing warnings/errors
    assert run_dialog.fail_msg_box.isVisible()
    show_warnings_button = run_dialog.show_warnings_button
    assert show_warnings_button.isEnabled()
    show_warnings_button.click()
    assert not run_dialog.fail_msg_box.isVisible()
    show_warnings_button.click()
    assert run_dialog.fail_msg_box.isVisible()


def test_denied_run_path_warning_dialog_releases_storage_lock(
    qtbot, opened_main_window_poly, use_tmpdir, monkeypatch
):
    # Populate runpath
    runpath = "poly_out/realization-0/iter-0"
    Path(runpath).mkdir(parents=True, exist_ok=True)
    Path(runpath).touch()

    # Open main window
    gui = opened_main_window_poly
    run_experiment_panel = wait_for_child(gui, qtbot, ExperimentPanel)

    # Mock class for experiment arguments
    class MockArgs:
        def __init__(self) -> None:
            self.mode = "ensemble_experiment"
            self.current_ensemble = "ensemble"
            self.experiment_name = "FooBar"

    monkeypatch.setattr(
        ExperimentPanel, "get_experiment_arguments", Mock(return_value=MockArgs())
    )

    # Mock the runpath warning window
    def mock_exec():
        # Assert the storage lock is initially locked
        assert run_experiment_panel._model._storage._lock.is_locked
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(
        QMessageBox,
        "exec",
        lambda _: mock_exec(),
    )

    run_experiment_panel.run_experiment()

    # Assert the storage lock has been unlocked
    assert not run_experiment_panel._model._storage._lock.is_locked


def test_that_summary_of_experiment_is_logged_when_running_poly_example_with_design_matrix(  # noqa: E501
    qtbot,
    copy_poly_case_with_design_matrix,
    caplog,
):
    caplog.set_level(logging.INFO)

    num_realizations = 5
    a_values = list(range(num_realizations))
    design_dict = {
        "REAL": list(range(num_realizations)),
        "a": a_values,
    }
    default_list = [["b", 1], ["c", 2]]
    copy_poly_case_with_design_matrix(design_dict, default_list)

    args = Mock()
    args.config = "poly.ert"

    with add_gui_log_handler() as log_handler:
        gui, *_ = ert.gui.main._start_initial_gui_window(args, log_handler)
        qtbot.addWidget(gui)

        experiment_panel = wait_for_child(gui, qtbot, ExperimentPanel)
        qtbot.wait_until(lambda: not experiment_panel.isHidden(), timeout=5000)

        @contextlib.contextmanager
        def mock_run_dialog():
            """Mocking run dialog and catching exceptions shaves off 2 seconds for this
            test, taking about 0.5 sec as a result"""
            original_init = RunDialog.__init__
            RunDialog.__init__ = Mock(return_value=None)
            try:
                yield
            finally:
                RunDialog.__init__ = original_init

        with contextlib.suppress(Exception), mock_run_dialog():
            experiment_panel.run_experiment()

        assert "Experiment summary:" in caplog.text
        assert "Runmodel: test_run" in caplog.text
        assert "Realizations: 1" in caplog.text
        assert "Parameters: 3" in caplog.text
        assert "Observations: 5" in caplog.text
