from threading import Thread
from PyQt4.QtCore import Qt, pyqtSignal, QTimer
from PyQt4.QtGui import QDialog, QVBoxLayout, QLayout, QMessageBox, QPushButton, QHBoxLayout, QColor, QLabel
from ert_gui.models.connectors.run import SimulationsTracker
from ert_gui.models.mixins.run_model import RunModelMixin
from ert_gui.widgets.legend import Legend
from ert_gui.widgets.progress import Progress
from ert_gui.widgets.simple_progress import SimpleProgress


class RunDialog(QDialog):

    simulationFinished = pyqtSignal()

    def __init__(self, run_model):
        QDialog.__init__(self)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.setWindowTitle("Simulations")

        assert isinstance(run_model, RunModelMixin)
        self.__run_model = run_model

        layout = QVBoxLayout()
        layout.setSizeConstraint(QLayout.SetFixedSize)

        self.simulations_tracker = SimulationsTracker()
        states = self.simulations_tracker.getList()

        self.total_progress = SimpleProgress()
        layout.addWidget(self.total_progress)

        self.progress = Progress()
        for state in states:
            self.progress.addState(state.state, QColor(*state.color), 100.0 * state.count / state.total_count)

        layout.addWidget(self.progress)

        legend_layout = QHBoxLayout()
        self.legends = {}
        for state in states:
            self.legends[state] = Legend("%s (%d/%d)", QColor(*state.color))
            self.legends[state].updateLegend(state.name, 0, 0)
            legend_layout.addWidget(self.legends[state])

        layout.addLayout(legend_layout)

        self.running_time = QLabel("")

        self.kill_button = QPushButton("Kill simulations")
        self.done_button = QPushButton("Done")
        self.done_button.setHidden(True)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.running_time)
        button_layout.addStretch()
        button_layout.addWidget(self.kill_button)
        button_layout.addWidget(self.done_button)

        layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.kill_button.clicked.connect(self.killJobs)
        self.done_button.clicked.connect(self.accept)
        self.simulationFinished.connect(self.hideKillAndShowDone)



        self.__updating = False
        self.__update_queued = False
        self.__simulation_started = False


    def startSimulation(self):
        simulation_thread = Thread(name="ert_gui_simulation_thread")
        simulation_thread.setDaemon(True)
        simulation_thread.run = self.__run_model.startSimulations
        simulation_thread.start()

        timer = QTimer(self)
        timer.setInterval(500)
        timer.timeout.connect(self.updateRunStatus)
        timer.start()


    def simulationDone(self):
        self.simulationFinished.emit()

    def updateRunStatus(self):
        self.total_progress.setProgress(self.__run_model.getProgress())

        total_count = self.__run_model.getQueueSize()
        queue_status = self.__run_model.getQueueStatus()
        states = self.simulations_tracker.getList()

        for state in states:
            state.count = 0
            state.total_count = total_count

        for state in states:
            for queue_state in queue_status:
                if queue_state in state.state:
                    state.count += queue_status[queue_state]

            self.progress.updateState(state.state, 100.0 * state.count / state.total_count)
            self.legends[state].updateLegend(state.name, state.count, state.total_count)

        self.setRunningTime()

        if self.__run_model.isFinished():
            self.hideKillAndShowDone()

    def setRunningTime(self):
        self.running_time.setText("Running time: %d seconds" % self.__run_model.getRunningTime())


    def killJobs(self):
        kill_job = QMessageBox.question(self, "Kill simulations?", "Are you sure you want to kill the currently running simulations?", QMessageBox.Yes | QMessageBox.No )

        if kill_job == QMessageBox.Yes:
            self.__run_model.killAllSimulations()
            QDialog.reject(self)

    def hideKillAndShowDone(self):
        self.kill_button.setHidden(True)
        self.done_button.setHidden(False)