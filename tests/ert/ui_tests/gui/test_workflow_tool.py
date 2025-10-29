import os
from collections.abc import Generator
from contextlib import contextmanager
from textwrap import dedent
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import Qt, QTimer

from ert.config import ErtConfig
from ert.gui.ertwidgets import ClosableDialog
from ert.gui.main import _setup_main_window
from ert.gui.main_window import ErtMainWindow
from ert.gui.tools.event_viewer import GUILogHandler
from ert.gui.tools.workflows import RunWorkflowWidget
from ert.plugins import get_site_plugins
from ert.run_models import EnsembleExperiment
from ert.storage import Storage

from .conftest import get_child, wait_for_child


@contextmanager
def _open_main_window(
    path,
) -> Generator[tuple[ErtMainWindow, Storage, ErtConfig], None, None]:
    (path / "config.ert").write_text(
        dedent("""
    QUEUE_SYSTEM LOCAL
    NUM_REALIZATIONS 1
    LOAD_WORKFLOW test_wf
    """)
    )
    (path / "test_wf").write_text("EXPORT_RUNPATH\n")
    config = ErtConfig.with_plugins(get_site_plugins()).from_file(path / "config.ert")

    args_mock = Mock()
    args_mock.config = "config.ert"
    # handler defined here to ensure lifetime until end of function, if inlined
    # it will cause the following error:
    # RuntimeError: wrapped C/C++ object of type GUILogHandler
    handler = GUILogHandler()
    gui = _setup_main_window(config, args_mock, handler, config.ens_path)
    yield gui
    gui.close()


@pytest.fixture
def open_gui(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        _open_main_window(tmp_path) as (gui),
    ):
        yield gui


def test_run_export_runpath_workflow(open_gui, qtbot, run_experiment):
    gui = open_gui
    run_experiment(EnsembleExperiment, gui)

    def handle_run_workflow_tool():
        dialog = wait_for_child(gui, qtbot, ClosableDialog)
        workflow_widget = get_child(dialog, RunWorkflowWidget)

        def close_all():
            if running_dlg := workflow_widget._running_workflow_dialog:
                running_dlg.accept()
            dialog.close()

        workflow_widget.workflowSucceeded.disconnect()
        workflow_widget.workflowSucceeded.connect(close_all)
        qtbot.mouseClick(workflow_widget.run_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(1000, handle_run_workflow_tool)
    gui.workflows_tool.trigger()

    assert os.path.isfile(".ert_runpath_list")


def test_run_workflow_with_no_ensemble_selected(qtbot, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.ert").write_text(
        dedent("""
            NUM_REALIZATIONS 1
            LOAD_WORKFLOW_JOB print_job PRINT
            LOAD_WORKFLOW print_workflow
            """)
    )

    print_job_content = "EXECUTABLE echo"
    (tmp_path / "print_job").write_text(print_job_content.strip())

    print_workflow_content = "PRINT Hello world"
    (tmp_path / "print_workflow").write_text(print_workflow_content.strip())

    ert_config = ErtConfig.from_file("config.ert")

    args_mock = Mock()
    args_mock.config = "config.ert"
    # handler defined here to ensure lifetime until end of function, if inlined
    # it will cause the following error:
    # RuntimeError: wrapped C/C++ object of type GUILogHandler
    handler = GUILogHandler()
    gui = _setup_main_window(ert_config, args_mock, handler, ert_config.ens_path)

    def handle_run_workflow_tool():
        dialog = wait_for_child(gui, qtbot, ClosableDialog)
        workflow_widget = get_child(dialog, RunWorkflowWidget)
        assert workflow_widget.notifier.current_ensemble is None
        assert workflow_widget.run_button.isEnabled()

        def close_all():
            if running_dlg := workflow_widget._running_workflow_dialog:
                running_dlg.accept()
            dialog.close()

        workflow_widget.workflowSucceeded.disconnect()
        workflow_widget.workflowSucceeded.connect(close_all)
        qtbot.mouseClick(workflow_widget.run_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(1000, handle_run_workflow_tool)
    gui.workflows_tool.trigger()
    assert capsys.readouterr().out == "Hello world\n"
    gui.close()
