from PyQt4.QtCore import Qt, pyqtSignal, QTimer
from PyQt4.QtGui import QDialog, QVBoxLayout, QLayout, QMessageBox, QPushButton, QHBoxLayout, QColor, QLabel
from ert_gui.models.connectors.run import SimulationRunner, SimulationsTracker
from ert_gui.widgets.legend import Legend
from ert_gui.widgets.progress import Progress
from ert_gui.widgets.simple_progress import SimpleProgress


class RunDialog(QDialog):

    simulationFinished = pyqtSignal()

    def __init__(self, simulation_runner):
        QDialog.__init__(self)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.setWindowTitle("Simulations")

        assert isinstance(simulation_runner, SimulationRunner)
        self.__simulation_runner = simulation_runner
        self.__simulation_runner.observable().attach(SimulationRunner.SIMULATION_FINISHED_EVENT, self.simulationDone)
        self.__simulation_runner.observable().attach(SimulationRunner.SIMULATION_PHASE_CHANGED_EVENT, self.statusChanged)


        layout = QVBoxLayout()
        layout.setSizeConstraint(QLayout.SetFixedSize)

        self.simulations_tracker = SimulationsTracker()
        self.simulations_tracker.observable().attach(SimulationsTracker.LIST_CHANGED_EVENT, self.statusChanged)
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

        timer = QTimer(self)
        timer.setInterval(500)
        timer.timeout.connect(self.setRunningTime)
        timer.start()

    def simulationDone(self):
        self.simulationFinished.emit()

    def setRunningTime(self):
        self.running_time.setText("Running time: %d seconds" % self.__simulation_runner.getRunningTime())

    def statusChanged(self):
        states = self.simulations_tracker.getList()

        for state in states:
            self.progress.updateState(state.state, 100.0 * state.count / state.total_count)
            self.legends[state].updateLegend(state.name, state.count, state.total_count)

        phase, phase_count = self.__simulation_runner.getTotalProgress()

        if phase < phase_count:
            progress = float(phase + self.simulations_tracker.getProgress()) / phase_count
        else:
            progress = 1.0

        self.total_progress.setProgress(progress)


    def killJobs(self):
        kill_job = QMessageBox.question(self, "Kill simulations?", "Are you sure you want to kill the currently running simulations?", QMessageBox.Yes | QMessageBox.No )

        if kill_job == QMessageBox.Yes:
            self.__simulation_runner.killAllSimulations()
            QDialog.reject(self)

    def hideKillAndShowDone(self):
        self.kill_button.setHidden(True)
        self.done_button.setHidden(False)