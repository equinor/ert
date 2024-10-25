import os
from contextlib import contextmanager
from textwrap import dedent
from typing import Generator, Tuple
from unittest.mock import Mock

import pytest
from qtpy.QtCore import Qt, QTimer

from ert.config import ErtConfig
from ert.gui.ertwidgets import ClosableDialog
from ert.gui.main import _setup_main_window
from ert.gui.main_window import ErtMainWindow
from ert.gui.tools.event_viewer import GUILogHandler
from ert.gui.tools.workflows import RunWorkflowWidget
from ert.plugins import ErtPluginContext
from ert.run_models import EnsembleExperiment
from ert.services import StorageService
from ert.storage import Storage, open_storage

from .conftest import get_child, wait_for_child


@contextmanager
def _open_main_window(
    path,
) -> Generator[Tuple[ErtMainWindow, Storage, ErtConfig], None, None]:
    (path / "config.ert").write_text(
        dedent("""
    QUEUE_SYSTEM LOCAL
    NUM_REALIZATIONS 1
    LOAD_WORKFLOW test_wf
    """)
    )
    (path / "test_wf").write_text("EXPORT_RUNPATH\n")
    with ErtPluginContext() as ctx:
        config = ErtConfig.with_plugins(
            ctx.plugin_manager.forward_model_steps
        ).from_file(path / "config.ert")

        args_mock = Mock()
        args_mock.config = "config.ert"
        # handler defined here to ensure lifetime until end of function, if inlined
        # it will cause the following error:
        # RuntimeError: wrapped C/C++ object of type GUILogHandler
        handler = GUILogHandler()
        with open_storage(config.ens_path, mode="w") as storage:
            gui = _setup_main_window(config, args_mock, handler, storage)
            yield gui, storage, config
            gui.close()


@pytest.fixture
def open_gui(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with _open_main_window(tmp_path) as (
        gui,
        _,
        config,
    ), StorageService.init_service(
        project=os.path.abspath(config.ens_path),
    ):
        yield gui


def test_run_export_runpath_workflow(open_gui, qtbot, run_experiment):
    gui = open_gui
    run_experiment(EnsembleExperiment, gui)

    def handle_run_workflow_tool():
        dialog = wait_for_child(gui, qtbot, ClosableDialog)
        workflow_widget = get_child(dialog, RunWorkflowWidget)

        def close_all():
            workflow_widget._running_workflow_dialog.accept()
            dialog.close()

        workflow_widget.workflowSucceeded.disconnect()
        workflow_widget.workflowSucceeded.connect(close_all)
        qtbot.mouseClick(workflow_widget.run_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(1000, handle_run_workflow_tool)
    gui.workflows_tool.trigger()

    assert os.path.isfile(".ert_runpath_list")
