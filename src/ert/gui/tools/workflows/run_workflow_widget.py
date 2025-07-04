from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QIcon, QMovie
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QToolButton,
    QWidget,
)

from _ert.threading import ErtThread
from ert.gui.ertwidgets import EnsembleSelector
from ert.gui.tools.workflows.workflow_dialog import WorkflowDialog
from ert.runpaths import Runpaths
from ert.workflow_runner import WorkflowRunner

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.gui.ertnotifier import ErtNotifier


class RunWorkflowWidget(QWidget):
    workflowSucceeded = Signal(list)
    workflowFailed = Signal()
    workflowKilled = Signal()

    def __init__(self, config: ErtConfig, notifier: ErtNotifier) -> None:
        self.config = config
        self.storage = notifier.storage
        self.notifier = notifier
        QWidget.__init__(self)

        layout = QFormLayout()

        self._workflow_combo = QComboBox()
        sorted_items = sorted(config.workflows.keys(), key=str.lower)
        self._workflow_combo.addItems(sorted_items)

        layout.addRow("Workflow", self._workflow_combo)

        self.source_ensemble_selector = EnsembleSelector(notifier, update_ert=False)
        layout.addRow("Ensemble", self.source_ensemble_selector)

        self.run_button = QToolButton()
        self.run_button.setIconSize(QSize(32, 32))
        self.run_button.setText("Start workflow")
        self.run_button.setIcon(QIcon("img:play_circle.svg"))
        self.run_button.clicked.connect(self.startWorkflow)
        self.run_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        layout.addRow(self.run_button)

        self.setLayout(layout)

        self._running_workflow_dialog: WorkflowDialog | None = None

        self.workflowSucceeded.connect(self.workflowFinished)
        self.workflowFailed.connect(self.workflowFinishedWithFail)
        self.workflowKilled.connect(self.workflowStoppedByUser)

        self._workflow_runner: WorkflowRunner | None = None

    def createSpinWidget(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout()

        size = 64
        spin_movie = QMovie("img:loading.gif")
        spin_movie.setSpeed(60)
        spin_movie.setScaledSize(QSize(size, size))
        spin_movie.start()

        processing_animation = QLabel()
        processing_animation.setMaximumSize(QSize(size, size))
        processing_animation.setMinimumSize(QSize(size, size))
        processing_animation.setMovie(spin_movie)
        layout.addWidget(processing_animation)

        processing_label = QLabel(
            f"Processing workflow '{self.getCurrentWorkflowName()}'"
        )
        layout.addWidget(processing_label, Qt.AlignmentFlag.AlignBottom)

        widget.setLayout(layout)

        return widget

    def cancelWorkflow(self) -> None:
        if self._workflow_runner is not None and self._workflow_runner.isRunning():
            cancel = QMessageBox.question(
                self,
                "Confirm cancel",
                "Are you sure you want to cancel the running workflow?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if cancel == QMessageBox.StandardButton.Yes:
                self._workflow_runner.cancel()
                if self._running_workflow_dialog is not None:
                    self._running_workflow_dialog.disableCloseButton()

    def getCurrentWorkflowName(self) -> str:
        index = self._workflow_combo.currentIndex()
        return sorted(self.config.workflows.keys(), key=str.lower)[index]

    def startWorkflow(self) -> None:
        logger.info("Gui utility: Run Workflow tool was used")
        dialog = WorkflowDialog("Running workflow", self.createSpinWidget(), self)
        dialog.closeButtonPressed.connect(self.cancelWorkflow)
        self._running_workflow_dialog = dialog

        workflow_thread = ErtThread(
            name="ert_gui_workflow_thread",
            target=self.runWorkflow,
            daemon=True,
            should_raise=False,
        )

        workflow = self.config.workflows[self.getCurrentWorkflowName()]
        ensemble = self.source_ensemble_selector.selected_ensemble

        self._workflow_runner = WorkflowRunner(
            workflow=workflow,
            fixtures={
                "ensemble": ensemble,
                "storage": self.storage,
                "random_seed": self.config.random_seed,
                "reports_dir": str(
                    self.config.analysis_config.log_path
                    / (ensemble.experiment.name if ensemble is not None else "")
                ),
                "observation_settings": self.config.analysis_config.observation_settings,  # noqa: E501
                "es_settings": self.config.analysis_config.es_settings,
                "run_paths": Runpaths(
                    jobname_format=self.config.runpath_config.jobname_format_string,
                    runpath_format=self.config.runpath_config.runpath_format_string,
                    filename=str(self.config.runpath_file),
                    substitutions=self.config.substitutions,
                    eclbase=self.config.runpath_config.eclbase_format_string,
                ),
            },
        )
        self._workflow_runner.run()

        workflow_thread.start()

        dialog.show()

    def runWorkflow(self) -> None:
        assert self._workflow_runner is not None
        while self._workflow_runner.isRunning():
            time.sleep(2)

        cancelled = self._workflow_runner.isCancelled()

        if cancelled:
            self.workflowKilled.emit()
        else:
            success = self._workflow_runner.workflowResult()

            if success:
                report = self._workflow_runner.workflowReport()
                failed_jobs = [k for k, v in report.items() if not v["completed"]]
                self.workflowSucceeded.emit(failed_jobs)
            else:
                self.workflowFailed.emit()

    def workflowFinished(self, failed_jobs: Iterable[str]) -> None:
        workflow_name = self.getCurrentWorkflowName()
        jobs_msg = "successfully!"
        if failed_jobs:
            jobs_msg = "\nThe following jobs failed: " + ", ".join(list(failed_jobs))

        QMessageBox.information(
            self,
            "Workflow completed!",
            f"The workflow '{workflow_name}' completed {jobs_msg}",
        )
        if self._running_workflow_dialog is not None:
            self._running_workflow_dialog.accept()
            self._running_workflow_dialog = None

    def workflowFinishedWithFail(self) -> None:
        assert self._workflow_runner is not None
        report = self._workflow_runner.workflowReport()
        failing_workflows = [
            (wfname, info) for wfname, info in report.items() if not info["completed"]
        ]

        title_text = f"Workflow{'s' if len(failing_workflows) > 1 else ''} failed"
        content_text = "\n\n".join(
            [
                f"{wfname} failed: \n {info['stderr'].strip()}"
                for wfname, info in failing_workflows
            ]
        )

        QMessageBox.critical(self, title_text, content_text)
        if self._running_workflow_dialog is not None:
            self._running_workflow_dialog.reject()
            self._running_workflow_dialog = None

    def workflowStoppedByUser(self) -> None:
        workflow_name = self.getCurrentWorkflowName()
        QMessageBox.information(
            self,
            "Workflow killed!",
            f"The workflow '{workflow_name}' was killed successfully!",
        )
        if self._running_workflow_dialog is not None:
            self._running_workflow_dialog.reject()
            self._running_workflow_dialog = None
