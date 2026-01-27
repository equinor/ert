import contextlib
import fileinput
import os
import os.path
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from textwrap import dedent
from typing import TypeVar
from unittest.mock import MagicMock, Mock

import pytest
from PyQt6.QtCore import QDir, Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QMessageBox,
    QPushButton,
    QToolButton,
    QWidget,
)
from pytestqt.qtbot import QtBot

from ert.base_model_context import use_runtime_plugins
from ert.config import ErtConfig
from ert.gui.ertwidgets import ClosableDialog, CreateExperimentDialog, EnsembleSelector
from ert.gui.main import ErtMainWindow, _setup_main_window, add_gui_log_handler
from ert.gui.simulation.experiment_panel import ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.view import RealizationWidget
from ert.gui.tools.load_results.load_results_panel import LoadResultsPanel
from ert.gui.tools.manage_experiments import ManageExperimentsPanel
from ert.gui.tools.manage_experiments.storage_widget import AddWidget, StorageWidget
from ert.plugins import get_site_plugins
from ert.run_models import EnsembleExperiment, MultipleDataAssimilation
from ert.storage import Storage
from tests.ert.unit_tests.gui.simulation.test_run_path_dialog import (
    handle_run_path_dialog,
)

DEFAULT_NUM_REALIZATIONS = 10


@pytest.fixture(scope="session", autouse=True)
def setup_svg_search_path():
    QDir.addSearchPath(
        "img", str(files("ert.gui").joinpath("../../ert/gui/resources/gui/img"))
    )


def open_gui_with_config(config_path) -> Iterator[ErtMainWindow]:
    with (
        _open_main_window(config_path) as (
            gui,
            _,
            __,
        ),
    ):
        yield gui


@pytest.fixture
def opened_main_window_poly(
    source_root, tmp_path, monkeypatch
) -> Iterator[ErtMainWindow]:
    monkeypatch.chdir(tmp_path)
    _new_poly_example(source_root, tmp_path)
    yield from open_gui_with_config(tmp_path / "poly.ert")


def _new_poly_example(
    source_root, destination, num_realizations: int = DEFAULT_NUM_REALIZATIONS
):
    shutil.copytree(
        os.path.join(source_root, "test-data", "ert", "poly_example"),
        destination,
        dirs_exist_ok=True,
    )

    with fileinput.input(destination / "poly.ert", inplace=True) as fin:
        for line in fin:
            if "NUM_REALIZATIONS" in line:
                # Speed of running tests dictates a low number of realizations,
                # while many realizations might be needed to observe
                # significant effect from update algorithms
                print(f"NUM_REALIZATIONS {num_realizations}", end="\n")
            elif "MAX_RUNNING" in line:
                print("MAX_RUNNING 2", end="\n")
            else:
                print(line, end="")


@contextmanager
def _open_main_window(path) -> Iterator[tuple[ErtMainWindow, Storage, ErtConfig]]:
    args_mock = Mock()
    args_mock.config = str(path)
    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        config = ErtConfig.with_plugins(site_plugins).from_file(path)
        with (
            add_gui_log_handler() as log_handler,
        ):
            gui = _setup_main_window(config, args_mock, log_handler, config.ens_path)
            yield gui, config.ens_path, config
            gui.close()


@pytest.fixture
def opened_main_window_minimal_realizations(source_root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _new_poly_example(source_root, tmp_path, 2)
    yield from open_gui_with_config(tmp_path / "poly.ert")


@pytest.fixture(scope="module")
def _esmda_run(run_experiment, source_root, tmp_path_factory):
    path = tmp_path_factory.mktemp("test-data")
    _new_poly_example(source_root, path)
    with (
        pytest.MonkeyPatch.context() as mp,
        _open_main_window(path / "poly.ert") as (
            gui,
            _,
            config,
        ),
    ):
        mp.chdir(path)
        run_experiment(MultipleDataAssimilation, gui)
        # Check that we produce update log
        log_paths = list(Path(config.analysis_config.log_path).iterdir())
        assert log_paths
        assert (log_paths[0] / "Report.report").exists()
        assert (log_paths[0] / "Report.csv").exists()

    return path


def _ensemble_experiment_run(
    run_experiment,
    source_root,
    tmp_path_factory,
    failing_reals: bool,
    with_templates: bool,
):
    path = tmp_path_factory.mktemp("test-data")
    _new_poly_example(source_root, path)
    if with_templates:
        (path / "foo.tmpl").write_text("template contents", encoding="utf-8")
        with (path / "poly.ert").open("a", encoding="utf-8") as f:
            f.write("RUN_TEMPLATE foo.tmpl foo.txt")
    with (
        pytest.MonkeyPatch.context() as mp,
        _open_main_window(path / "poly.ert") as (
            gui,
            _,
            _,
        ),
    ):
        mp.chdir(path)
        if failing_reals:
            Path("poly_eval.py").write_text(
                dedent(
                    """\
                        #!/usr/bin/env python3
                        import os
                        import sys
                        import json

                        def _load_coeffs(filename):
                            with open(filename, encoding="utf-8") as f:
                                return json.load(f)

                        def _evaluate(coeffs, x):
                            return (coeffs["a"]["value"] * x**2 +
                                    coeffs["b"]["value"] * x + coeffs["c"]["value"])

                        if __name__ == "__main__":
                            if int(os.getenv("_ERT_REALIZATION_NUMBER")) % 2 == 0:
                                sys.exit(1)
                            coeffs = _load_coeffs("parameters.json")
                            output = [_evaluate(coeffs, x) for x in range(10)]
                            with open("poly.out", "w", encoding="utf-8") as f:
                                f.write("\\n".join(map(str, output)))
                        """
                ),
                encoding="utf-8",
            )

            os.chmod(
                "poly_eval.py",
                os.stat("poly_eval.py").st_mode
                | stat.S_IXUSR
                | stat.S_IXGRP
                | stat.S_IXOTH,
            )
        run_experiment(EnsembleExperiment, gui)

    return path


@pytest.fixture
def esmda_has_run(_esmda_run, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(_esmda_run, tmp_path, dirs_exist_ok=True)
    with (
        _open_main_window(tmp_path / "poly.ert") as (
            gui,
            _,
            __,
        ),
    ):
        yield gui


@pytest.fixture
def ensemble_experiment_has_run(
    tmp_path, monkeypatch, run_experiment, source_root, tmp_path_factory
):
    yield from _ensemble_experiment_has_run(
        tmp_path,
        monkeypatch,
        run_experiment,
        source_root,
        tmp_path_factory,
        failing_reals=True,
        with_templates=False,
    )


@pytest.fixture
def ensemble_experiment_has_run_no_failure(
    tmp_path, monkeypatch, run_experiment, source_root, tmp_path_factory
):
    yield from _ensemble_experiment_has_run(
        tmp_path,
        monkeypatch,
        run_experiment,
        source_root,
        tmp_path_factory,
        failing_reals=False,
        with_templates=True,
    )


def _ensemble_experiment_has_run(
    tmp_path,
    monkeypatch,
    run_experiment,
    source_root,
    tmp_path_factory,
    failing_reals,
    with_templates,
):
    monkeypatch.chdir(tmp_path)
    test_files = _ensemble_experiment_run(
        run_experiment, source_root, tmp_path_factory, failing_reals, with_templates
    )
    shutil.copytree(test_files, tmp_path, dirs_exist_ok=True)
    yield from open_gui_with_config(tmp_path / "poly.ert")


@pytest.fixture(name="run_experiment", scope="module")
def run_experiment_fixture(request):
    def func(experiment_mode, gui, click_done=True):
        qtbot = QtBot(request)
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree("poly_out")
        # Select correct experiment in the simulation panel
        experiment_panel = get_child(gui, ExperimentPanel)
        simulation_mode_combo = get_child(experiment_panel, QComboBox)
        simulation_mode_combo.setCurrentText(experiment_mode.display_name())
        simulation_settings = experiment_panel._experiment_widgets[
            experiment_panel.get_current_experiment_type()
        ]
        if hasattr(simulation_settings, "_ensemble_name_field"):
            simulation_settings._ensemble_name_field.setText("iter-0")

        # Click start simulation and agree to the message
        run_experiment = get_child(experiment_panel, QWidget, name="run_experiment")

        def handle_dialog():
            QTimer.singleShot(
                500,
                lambda: handle_run_path_dialog(gui, qtbot, delete_run_path=False),
            )

        if experiment_mode.name() not in {
            "Ensemble experiment",
            "Evaluate ensemble",
        }:
            QTimer.singleShot(500, handle_dialog)
        qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)

        if click_done:
            # The Run dialog opens, click show details and wait until done appears
            # then click it
            run_dialog = wait_for_child(gui, qtbot, RunDialog, timeout=10000)
            qtbot.waitUntil(
                lambda: run_dialog.is_simulation_done() is True, timeout=200000
            )
            qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

            # Assert that the number of boxes in the detailed view is
            # equal to the number of realizations
            realization_widget = run_dialog._tab_widget.currentWidget()
            assert isinstance(realization_widget, RealizationWidget)
            list_model = realization_widget._real_view.model()
            assert (
                list_model.rowCount()
                == experiment_panel.config.runpath_config.num_realizations
            )

    return func


@pytest.fixture(name="active_realizations")
def active_realizations_fixture() -> Mock:
    active_reals = MagicMock()
    active_reals.count = Mock(return_value=10)
    active_reals.__iter__.return_value = [True] * 10
    return active_reals


def load_results_manually(qtbot, gui, ensemble_name="default"):
    def handle_load_results_dialog():
        dialog = wait_for_child(gui, qtbot, ClosableDialog)
        panel = get_child(dialog, LoadResultsPanel)

        ensemble_selector = get_child(panel, EnsembleSelector)
        index = ensemble_selector.findText(ensemble_name, Qt.MatchFlag.MatchContains)
        assert index != -1
        ensemble_selector.setCurrentIndex(index)

        # click on "Load"
        load_button = get_child(panel.parent(), QPushButton, name="Load")

        # Verify that the messagebox is the success kind
        def handle_popup_dialog():
            messagebox = QApplication.activeModalWidget()
            assert isinstance(messagebox, QMessageBox)
            assert messagebox.text() == "Successfully loaded all realizations"
            ok_button = messagebox.button(QMessageBox.StandardButton.Ok)
            qtbot.mouseClick(ok_button, Qt.MouseButton.LeftButton)

        QTimer.singleShot(2000, handle_popup_dialog)
        qtbot.mouseClick(load_button, Qt.MouseButton.LeftButton)
        dialog.close()

    QTimer.singleShot(1000, handle_load_results_dialog)
    gui.load_results_tool.trigger()


def add_experiment_in_manage_experiment_dialog(
    qtbot,
    manage_experiment_dialog,
    experiment_name="My_experiment",
    ensemble_name="default",
):
    current_tab = manage_experiment_dialog.currentWidget()
    assert current_tab.objectName() == "create_new_ensemble_tab"
    storage_widget = get_child(current_tab, StorageWidget)

    def handle_add_dialog():
        dialog = wait_for_child(current_tab, qtbot, CreateExperimentDialog)
        dialog._experiment_edit.setText(experiment_name)
        dialog._ensemble_edit.setText(ensemble_name)
        qtbot.mouseClick(dialog._ok_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(1000, handle_add_dialog)
    add_widget = get_child(storage_widget, AddWidget)
    qtbot.mouseClick(add_widget.addButton, Qt.MouseButton.LeftButton)


def add_experiment_manually(
    qtbot, gui, experiment_name="My_experiment", ensemble_name="default"
):
    button_manage_experiments = get_child(
        gui, QToolButton, name="button_Manage_experiments"
    )
    qtbot.mouseClick(button_manage_experiments, Qt.MouseButton.LeftButton)
    experiments_panel = get_child(gui, ManageExperimentsPanel)

    # Open the create new experiment tab
    experiments_panel.setCurrentIndex(0)
    add_experiment_in_manage_experiment_dialog(
        qtbot, experiments_panel, experiment_name, ensemble_name
    )
    experiments_panel.close()


V = TypeVar("V")


def wait_for_child(gui, qtbot: QtBot, typ: type[V], timeout=5000, **kwargs) -> V:
    qtbot.waitUntil(lambda: gui.findChild(typ) is not None, timeout=timeout)
    return get_child(gui, typ, **kwargs)


def get_child(gui: QWidget, typ: type[V], *args, **kwargs) -> V:
    child = gui.findChild(typ, *args, **kwargs)
    assert isinstance(child, typ)
    return child


def get_children(gui: QWidget, typ: type[V], *args, **kwargs) -> list[V]:
    children: list[typ] = gui.findChildren(typ, *args, **kwargs)
    return children
