import time
from threading import Thread

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QToolButton,
    QWidget,
)

from ert._c_wrappers.job_queue import WorkflowRunner
from ert.gui.ertwidgets import addHelpToWidget, resourceIcon, resourceMovie
from ert.gui.tools.workflows.workflow_dialog import WorkflowDialog


class RunWorkflowWidget(QWidget):

    workflowSucceeded = Signal(list)
    workflowFailed = Signal()
    workflowKilled = Signal()

    def __init__(self, ert):
        self.ert = ert
        QWidget.__init__(self)

        layout = QHBoxLayout()
        layout.addSpacing(10)

        self._workflow_combo = QComboBox()
        addHelpToWidget(self._workflow_combo, "run/workflow")

        self._workflow_combo.addItems(
            sorted(ert.getWorkflowList().getWorkflowNames(), key=str.lower)
        )

        layout.addWidget(QLabel("Select workflow:"), 0, Qt.AlignVCenter)
        layout.addWidget(self._workflow_combo, 0, Qt.AlignVCenter)

        # simulation_mode_layout.addStretch()
        layout.addSpacing(20)

        self.run_button = QToolButton()
        self.run_button.setIconSize(QSize(32, 32))
        self.run_button.setText("Start workflow")
        self.run_button.setIcon(resourceIcon("play_circle.svg"))
        self.run_button.clicked.connect(self.startWorkflow)
        self.run_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        layout.addWidget(self.run_button)
        layout.addStretch(1)

        self.setLayout(layout)

        self._running_workflow_dialog = None

        self.workflowSucceeded.connect(self.workflowFinished)
        self.workflowFailed.connect(self.workflowFinishedWithFail)
        self.workflowKilled.connect(self.workflowStoppedByUser)

        self._workflow_runner = None
        """:type: WorkflowRunner"""

    def createSpinWidget(self):
        widget = QWidget()
        layout = QHBoxLayout()

        size = 64
        spin_movie = resourceMovie("loading.gif")
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
        layout.addWidget(processing_label, Qt.AlignBottom)

        widget.setLayout(layout)

        return widget

    def cancelWorkflow(self):
        if self._workflow_runner.isRunning():
            cancel = QMessageBox.question(
                self,
                "Confirm cancel",
                "Are you sure you want to cancel the running workflow?",
                QMessageBox.Yes | QMessageBox.No,
            )

            if cancel == QMessageBox.Yes:
                self._workflow_runner.cancel()
                self._running_workflow_dialog.disableCloseButton()

    def getCurrentWorkflowName(self):
        index = self._workflow_combo.currentIndex()
        return sorted(self.ert.getWorkflowList().getWorkflowNames(), key=str.lower)[
            index
        ]

    def startWorkflow(self):
        self._running_workflow_dialog = WorkflowDialog(
            "Running workflow", self.createSpinWidget(), self
        )
        self._running_workflow_dialog.closeButtonPressed.connect(self.cancelWorkflow)

        workflow_thread = Thread(name="ert_gui_workflow_thread")
        workflow_thread.daemon = True
        workflow_thread.run = self.runWorkflow

        workflow_list = self.ert.getWorkflowList()

        workflow = workflow_list[self.getCurrentWorkflowName()]
        context = workflow_list.getContext()
        self._workflow_runner = WorkflowRunner(workflow, self.ert, context)
        self._workflow_runner.run()

        workflow_thread.start()

        self._running_workflow_dialog.show()

    def runWorkflow(self):
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

    def workflowFinished(self, failed_jobs):
        workflow_name = self.getCurrentWorkflowName()
        jobs_msg = "successfully!"
        if failed_jobs:
            jobs_msg = "\nThe following jobs failed: " + ", ".join(list(failed_jobs))

        QMessageBox.information(
            self,
            "Workflow completed!",
            f"The workflow '{workflow_name}' completed {jobs_msg}",
        )
        self._running_workflow_dialog.accept()
        self._running_workflow_dialog = None

    def workflowFinishedWithFail(self):
        workflow_name = self.getCurrentWorkflowName()

        error = self._workflow_runner.workflowError()

        QMessageBox.critical(
            self,
            "Workflow failed!",
            f"The workflow '{workflow_name}' failed!\n\n{error}",
        )
        self._running_workflow_dialog.reject()
        self._running_workflow_dialog = None

    def workflowStoppedByUser(self):
        workflow_name = self.getCurrentWorkflowName()
        QMessageBox.information(
            self,
            "Workflow killed!",
            f"The workflow '{workflow_name}' was killed successfully!",
        )
        self._running_workflow_dialog.reject()
        self._running_workflow_dialog = None
