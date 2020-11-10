from threading import Thread

from qtpy.QtCore import Qt, QTimer, QSize, Signal, Slot
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from ecl.util.util import BoolVector
from ert_gui.ertwidgets import Legend, resourceMovie
from ert_gui.simulation import DetailedProgressWidget, Progress, SimpleProgress
from ert_gui.tools.plot.plot_tool import PlotTool
from ert_shared.models import BaseRunModel
from ert_shared.tracker.events import DetailedEvent, EndEvent, GeneralEvent
from ert_shared.tracker.factory import create_tracker
from ert_shared.tracker.utils import format_running_time
from res.job_queue import JobStatusType


class RunDialog(QDialog):
    simulation_done = Signal(bool, str)

    def __init__(self, config_file, run_model, simulation_arguments, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowFlags(Qt.Window)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModal)
        self.setWindowTitle("Simulations - {}".format(config_file))

        self._run_model = run_model

        ert = None
        if isinstance(run_model, BaseRunModel):
            ert = run_model.ert()

        self._simulations_argments = simulation_arguments
        self.simulations_tracker = create_tracker(
            run_model, qtimer_cls=QTimer,
            event_handler=self._on_tracker_event,
            num_realizations=self._simulations_argments["active_realizations"].count())

        self._ticker = QTimer(self)
        self._ticker.timeout.connect(self._on_ticker)

        states = self.simulations_tracker.get_states()
        self.state_colors = {state.name: state.color for state in states}
        self.state_colors['Success'] = self.state_colors["Finished"]
        self.state_colors['Failure'] = self.state_colors["Failed"]

        self.total_progress = SimpleProgress()

        status_layout = QHBoxLayout()
        status_layout.addStretch()
        self.__status_label = QLabel()
        status_layout.addWidget(self.__status_label)
        status_layout.addStretch()
        status_widget_container = QWidget()
        status_widget_container.setLayout(status_layout)

        self.progress = Progress()
        self.progress.setIndeterminateColor(self.total_progress.color)
        for state in states:
            self.progress.addState(state.state, QColor(*state.color), 100.0 * state.count / state.total_count)

        legend_layout = QHBoxLayout()
        self.legends = {}
        for state in states:
            self.legends[state] = Legend("%s (%d/%d)", QColor(*state.color))
            self.legends[state].updateLegend(state.name, 0, 0)
            legend_layout.addWidget(self.legends[state])

        legend_widget_container = QWidget()
        legend_widget_container.setLayout(legend_layout)

        self.running_time = QLabel("")

        self.plot_tool = PlotTool(config_file)
        self.plot_tool.setParent(self)
        self.plot_button = QPushButton(self.plot_tool.getName())
        self.plot_button.clicked.connect(self.plot_tool.trigger)
        self.plot_button.setEnabled(ert is not None)

        self.kill_button = QPushButton("Kill simulations")
        self.done_button = QPushButton("Done")
        self.done_button.setHidden(True)
        self.restart_button = QPushButton("Restart")
        self.restart_button.setHidden(True)
        self.show_details_button = QPushButton("Details")
        self.show_details_button.setCheckable(True)

        size = 20
        spin_movie = resourceMovie("ide/loading.gif")
        spin_movie.setSpeed(60)
        spin_movie.setScaledSize(QSize(size, size))
        spin_movie.start()

        self.processing_animation = QLabel()
        self.processing_animation.setMaximumSize(QSize(size, size))
        self.processing_animation.setMinimumSize(QSize(size, size))
        self.processing_animation.setMovie(spin_movie)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.processing_animation)
        button_layout.addWidget(self.running_time)
        button_layout.addStretch()
        button_layout.addWidget(self.show_details_button)
        button_layout.addWidget(self.plot_button)
        button_layout.addWidget(self.kill_button)
        button_layout.addWidget(self.done_button)
        button_layout.addWidget(self.restart_button)
        button_widget_container = QWidget()
        button_widget_container.setLayout(button_layout)

        self.detailed_progress = DetailedProgressWidget(self, self.state_colors)
        self.detailed_progress.setVisible(False)
        self.dummy_widget_container = QWidget() #Used to keep the other widgets from stretching

        layout = QVBoxLayout()
        layout.addWidget(self.total_progress)
        layout.addWidget(status_widget_container)
        layout.addWidget(self.progress)
        layout.addWidget(legend_widget_container)
        layout.addWidget(self.detailed_progress)
        layout.addWidget(self.dummy_widget_container)
        layout.addWidget(button_widget_container)

        layout.setStretch(0, 0)
        layout.setStretch(1, 0)
        layout.setStretch(2, 0)
        layout.setStretch(3, 0)
        layout.setStretch(4, 1)
        layout.setStretch(5, 1)
        layout.setStretch(6, 0)

        self.setLayout(layout)

        self.kill_button.clicked.connect(self.killJobs)
        self.done_button.clicked.connect(self.accept)
        self.restart_button.clicked.connect(self.restart_failed_realizations)
        self.show_details_button.clicked.connect(self.toggle_detailed_progress)
        self.simulation_done.connect(self._on_simulation_done)

    def reject(self):
        return

    def closeEvent(self, QCloseEvent):
        self.simulations_tracker.stop()
        if self._run_model.isFinished():
            self.simulation_done.emit(self._run_model.hasRunFailed(),
                                      self._run_model.getFailMessage())
        else:
            # Kill jobs if dialog is closed
            if self.killJobs() != QMessageBox.Yes:
                QCloseEvent.ignore()

    def startSimulation(self):
        self._run_model.reset()
        self.simulations_tracker.reset()

        def run():
            self._run_model.startSimulations( self._simulations_argments )

        simulation_thread = Thread(name="ert_gui_simulation_thread")
        simulation_thread.setDaemon(True)
        simulation_thread.run = run
        simulation_thread.start()

        self._ticker.start(1000)
        self.simulations_tracker.track()

    def killJobs(self):

        msg =  "Are you sure you want to kill the currently running simulations?"
        if self._run_model.getQueueStatus().get(JobStatusType.JOB_QUEUE_UNKNOWN, 0) > 0:
            msg += "\n\nKilling a simulation with unknown status will not kill the realizations already submitted!"
        kill_job = QMessageBox.question(self, "Kill simulations?",msg, QMessageBox.Yes | QMessageBox.No )

        if kill_job == QMessageBox.Yes:
            if self.simulations_tracker.request_termination():
                self.reject()
        return kill_job

    @Slot(bool, str)
    def _on_simulation_done(self, failed, failed_msg):
        self.simulations_tracker.stop()
        self.processing_animation.hide()
        self.kill_button.setHidden(True)
        self.done_button.setHidden(False)
        self.restart_button.setVisible(self.has_failed_realizations())
        self.restart_button.setEnabled(self._run_model.support_restart)

        if failed:
            QMessageBox.critical(self, "Simulations failed!",
                                 "The simulation failed with the following " +
                                 "error:\n\n{}".format(failed_msg))

    @Slot()
    def _on_ticker(self):
        runtime = self._run_model.get_runtime()
        self.running_time.setText(format_running_time(runtime))

    @Slot(object)
    def _on_tracker_event(self, event):
        if isinstance(event, GeneralEvent):
            self.total_progress.setProgress(event.progress)
            self.progress.setIndeterminate(event.indeterminate)

            if event.indeterminate:
                for state in event.sim_states:
                    self.legends[state].updateLegend(state.name, 0, 0)
            else:
                for state in event.sim_states:
                    try:
                        self.progress.updateState(
                            state.state, 100.0 * state.count / state.total_count)
                    except ZeroDivisionError:
                        # total_count not set by some slow tracker (EE)
                        pass
                    self.legends[state].updateLegend(
                        state.name, state.count, state.total_count)

        if isinstance(event, DetailedEvent):
            if not self.progress.get_indeterminate():
                self.detailed_progress.set_progress(event.details,
                                                    event.iteration)

        if isinstance(event, EndEvent):
            self.simulation_done.emit(event.failed, event.failed_msg)

    def has_failed_realizations(self):
        completed = self._run_model.completed_realizations_mask
        initial = self._run_model.initial_realizations_mask
        for (index, successful) in enumerate(completed):
            if initial[index] and not successful:
                return True
        return False


    def count_successful_realizations(self):
        """
        Counts the realizations completed in the prevoius ensemble run
        :return:
        """
        completed = self._run_model.completed_realizations_mask
        return completed.count(True)

    def create_mask_from_failed_realizations(self):
        """
        Creates a BoolVector mask representing the failed realizations
        :return: Type BoolVector
        """
        completed = self._run_model.completed_realizations_mask
        initial = self._run_model.initial_realizations_mask
        inverted_mask = BoolVector(  default_value = False )
        for (index, successful) in enumerate(completed):
            inverted_mask[index] = initial[index] and not successful
        return inverted_mask


    def restart_failed_realizations(self):

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setText("Note that workflows will only be executed on the restarted realizations and that this might have unexpected consequences.")
        msg.setWindowTitle("Restart Failed Realizations")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        result = msg.exec_()

        if result == QMessageBox.Ok:
            self.restart_button.setVisible(False)
            self.kill_button.setVisible(True)
            self.done_button.setVisible(False)
            active_realizations = self.create_mask_from_failed_realizations()
            self._simulations_argments['active_realizations'] = active_realizations
            self._simulations_argments['prev_successful_realizations'] = self._simulations_argments.get('prev_successful_realizations', 0)
            self._simulations_argments['prev_successful_realizations'] += self.count_successful_realizations()
            self.startSimulation()



    def toggle_detailed_progress(self):

        self.detailed_progress.setVisible(not(self.detailed_progress.isVisible()))
        self.dummy_widget_container.setVisible(not(self.detailed_progress.isVisible()))
        self.adjustSize()
