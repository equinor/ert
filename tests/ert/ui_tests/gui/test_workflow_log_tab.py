from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QToolButton
from pytestqt.qtbot import QtBot

from ert.gui.experiments import ExperimentPanel, RunDialog
from ert.gui.experiments.ensemble_experiment_panel import EnsembleExperimentPanel
from ert.gui.experiments.view import WorkflowLogWidget
from ert.run_models import EnsembleExperiment

from .conftest import open_gui_with_config

_FORWARD_MODEL_SCRIPT = "#!/bin/sh\nexit 0\n"


def _write_common_config_files() -> None:
    forward_model_script = Path("forward_model.sh")
    forward_model_script.write_text(_FORWARD_MODEL_SCRIPT, encoding="utf-8")
    forward_model_script.chmod(0o755)
    Path("FORWARD_MODEL_JOB").write_text(
        "EXECUTABLE forward_model.sh\n", encoding="utf-8"
    )


@contextmanager
def _run_experiment(qtbot: QtBot, config_file: str) -> Iterator[RunDialog]:
    with open_gui_with_config(config_file) as gui:
        experiment_panel = gui.findChild(ExperimentPanel)
        assert experiment_panel
        simulation_mode_combo = experiment_panel.findChild(QComboBox)
        assert simulation_mode_combo
        simulation_mode_combo.setCurrentText(EnsembleExperiment.name())
        simulation_settings = gui.findChild(EnsembleExperimentPanel)
        assert simulation_settings
        simulation_settings._experiment_name_field.setText("workflow_log_experiment")

        run_experiment = experiment_panel.findChild(QToolButton, name="run_experiment")
        assert run_experiment
        qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)

        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None, timeout=10000)
        run_dialog = gui.findChild(RunDialog)
        assert run_dialog
        qtbot.waitUntil(run_dialog.is_experiment_done, timeout=60000)
        yield run_dialog


def _workflow_log_widget(run_dialog: RunDialog) -> WorkflowLogWidget | None:
    for i in range(run_dialog._tab_widget.count()):
        widget = run_dialog._tab_widget.widget(i)
        if isinstance(widget, WorkflowLogWidget):
            return widget
    return None


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.timeout(120)
def test_that_hooked_workflow_output_is_shown_in_workflows_tab(
    qtbot: QtBot,
) -> None:
    _write_common_config_files()

    workflow_script = Path("printing_workflow.py")
    workflow_script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('greetings from the workflow')\n"
        "print('a warning', file=sys.stderr)\n",
        encoding="utf-8",
    )
    workflow_script.chmod(0o755)
    Path("PRINTING_WORKFLOW_JOB").write_text(
        "EXECUTABLE printing_workflow.py\n", encoding="utf-8"
    )

    Path("config.ert").write_text(
        """NUM_REALIZATIONS 1
QUEUE_SYSTEM LOCAL
INSTALL_JOB forward_model FORWARD_MODEL_JOB
FORWARD_MODEL forward_model
LOAD_WORKFLOW_JOB PRINTING_WORKFLOW_JOB PRINTING_WORKFLOW
HOOK_WORKFLOW_JOB printing_job PRINTING_WORKFLOW POST_SIMULATION
""",
        encoding="utf-8",
    )

    with _run_experiment(qtbot, "config.ert") as run_dialog:
        widget = _workflow_log_widget(run_dialog)
        assert widget is not None, "no Workflows tab was added to the run dialog"
        assert widget._table.rowCount() == 1
        assert widget._table.item(0, 0).text() == "POST_SIMULATION"
        assert widget._table.item(0, 2).text() == "PRINTING_WORKFLOW"
        assert widget._table.item(0, 3).text() == "Succeeded"

        widget._table.selectRow(0)
        assert "greetings from the workflow" in widget._stdout_view.toPlainText()
        assert "a warning" in widget._stderr_view.toPlainText()


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.timeout(120)
def test_that_no_workflows_tab_is_added_when_experiment_has_no_workflows(
    qtbot: QtBot,
) -> None:
    _write_common_config_files()

    Path("config.ert").write_text(
        """NUM_REALIZATIONS 1
QUEUE_SYSTEM LOCAL
INSTALL_JOB forward_model FORWARD_MODEL_JOB
FORWARD_MODEL forward_model
""",
        encoding="utf-8",
    )

    with _run_experiment(qtbot, "config.ert") as run_dialog:
        assert _workflow_log_widget(run_dialog) is None
