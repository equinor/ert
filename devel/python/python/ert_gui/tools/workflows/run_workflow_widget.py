from threading import Thread
from PyQt4.QtCore import QSize, Qt, pyqtSignal
from PyQt4.QtGui import QWidget, QHBoxLayout, QLabel, QToolButton, QMovie, QVBoxLayout, QMessageBox
import time
import sys
from ert_gui.models.connectors.run import WorkflowsModel
from ert_gui.widgets import util
from ert_gui.widgets.combo_choice import ComboChoice
from ert_gui.widgets.confirm_dialog import ConfirmDialog
from ert_gui.widgets.workflow_dialog import WorkflowDialog


class RunWorkflowWidget(QWidget):

    workflowSucceeded = pyqtSignal()
    workflowFailed = pyqtSignal()
    workflowKilled = pyqtSignal()
    reloadErtTriggered = pyqtSignal()

    def __init__(self):
        QWidget.__init__(self)

        layout = QHBoxLayout()
        layout.addSpacing(10)

        workflow_model = WorkflowsModel()

        # workflow_model.observable().attach(WorkflowsModel.CURRENT_CHOICE_CHANGED_EVENT, self.showWorkflow)
        workflow_combo = ComboChoice(workflow_model, "Select Workflow", "run/workflow")
        layout.addWidget(QLabel(workflow_combo.getLabel()), 0, Qt.AlignVCenter)
        layout.addWidget(workflow_combo, 0, Qt.AlignVCenter)

        # simulation_mode_layout.addStretch()
        layout.addSpacing(20)

        self.run_button = QToolButton()
        self.run_button.setIconSize(QSize(32, 32))
        self.run_button.setText("Start Workflow")
        self.run_button.setIcon(util.resourceIcon("ide/gear_in_play"))
        self.run_button.clicked.connect(self.startWorkflow)
        self.run_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        layout.addWidget(self.run_button)
        layout.addStretch(1)

        self.setLayout(layout)

        self.__running_workflow_dialog = None
        self.__confirm_stop_dialog = None

        self.workflowSucceeded.connect(self.workflowFinished)
        self.workflowFailed.connect(self.workflowFinishedWithFail)
        self.workflowKilled.connect(self.workflowStopedByUser)

        self.__workflow_runner = None


    def createSpinWidget(self):
        widget = QWidget()
        layout = QHBoxLayout()

        size = 64
        spin_movie = util.resourceMovie("ide/loading.gif")
        spin_movie.setSpeed(60)
        spin_movie.setScaledSize(QSize(size, size))
        spin_movie.start()

        processing_animation = QLabel()
        processing_animation.setMaximumSize(QSize(size, size))
        processing_animation.setMinimumSize(QSize(size, size))
        processing_animation.setMovie(spin_movie)
        layout.addWidget(processing_animation)

        processing_label = QLabel("Processing workflow '%s'" % WorkflowsModel().getCurrentChoice())
        layout.addWidget(processing_label, Qt.AlignBottom)

        widget.setLayout(layout)

        return widget

    def confirmCancelWorkflow(self):
        self.__confirm_stop_dialog = ConfirmDialog("Confirm stop","Ert might need to shut down when a workflow is stopped - do you want to proceed?",self)
        self.__confirm_stop_dialog.confirmButtonPressed.connect(self.cancelWorkflow)
        self.__confirm_stop_dialog.show()


    def cancelWorkflow(self):
        if self.__workflow_runner.isRunning:
            self.__workflow_runner.cancelWorkflow()
            self.__running_workflow_dialog.accept()
            self.__confirm_stop_dialog.accept()
            if not self.__workflow_runner.isExternalWorkflow():
                #internal workflow we need to shutdown ERT to continue
                sys.exit(0)

    def startWorkflow(self):
        self.__running_workflow_dialog = WorkflowDialog("Running Workflow", self.createSpinWidget(), self)
        #self.__running_workflow_dialog.disableCloseButton()
        self.__running_workflow_dialog.closeButtonPressed.connect(self.confirmCancelWorkflow)

        workflow_thread = Thread(name="ert_gui_workflow_thread")
        workflow_thread.setDaemon(True)
        workflow_thread.run = self.runWorkflow

        self.__workflow_runner = WorkflowsModel().createWorkflowRunner()
        self.__workflow_runner.runWorkflow()

        workflow_thread.start()

        self.__running_workflow_dialog.show()

    def runWorkflow(self):
        while self.__workflow_runner.isRunning():
            time.sleep(2)

        killed = self.__workflow_runner.isKilled()
      
        if killed:
            self.workflowKilled.emit()
        else:
            success = self.__workflow_runner.getWorkflowResult()

            if not success:
                self.workflowFailed.emit()
            else:
                self.workflowSucceeded.emit()

    def workflowFinished(self):
        workflow_name = WorkflowsModel().getCurrentChoice()
        QMessageBox.information(self, "Workflow completed!", "The workflow '%s' completed successfully!" % workflow_name)
        self.__running_workflow_dialog.accept()
        self.__running_workflow_dialog = None

    def workflowFinishedWithFail(self):
        workflow_name = WorkflowsModel().getCurrentChoice()
        error = WorkflowsModel().getError()
        QMessageBox.critical(self, "Workflow failed!", "The workflow '%s' failed!\n\n%s" % (workflow_name, error))
        self.__running_workflow_dialog.reject()
        self.__running_workflow_dialog = None

    def workflowStopedByUser(self):
        workflow_name = WorkflowsModel().getCurrentChoice()
        QMessageBox.information(self, "Workflow killed!", "The workflow '%s' was killed successfully!" % workflow_name)
        self.__running_workflow_dialog = None