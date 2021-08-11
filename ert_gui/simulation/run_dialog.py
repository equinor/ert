import asyncio
from threading import Thread
from PyQt5.QtWidgets import QAbstractItemView

from ecl.util.util import BoolVector
from ert_gui.ertwidgets import resourceMovie
from ert_gui.model.job_list import JobListProxyModel
from ert_gui.model.snapshot import RealIens, SnapshotModel, FileRole
from ert_gui.simulation.tracker_worker import TrackerWorker
from ert_gui.tools.file import FileDialog
from ert_gui.tools.plot.plot_tool import PlotTool
from ert_shared.models import BaseRunModel
from ert_shared.status.entity.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert_shared.status.tracker.factory import create_tracker
from ert_shared.status.utils import format_running_time
from qtpy.QtCore import QModelIndex, QSize, Qt, QThread, QTimer, Signal, Slot
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QHeaderView,
    QProgressBar,
)
from res.job_queue import JobStatusType
from ert_gui.simulation.view.progress import ProgressView
from ert_gui.simulation.view.legend import LegendView
from ert_gui.simulation.view.realization import RealizationWidget
from ert_gui.model.progress_proxy import ProgressProxyModel


_TOTAL_PROGRESS_TEMPLATE = "Total progress {total_progress}% — {phase_name}"


class RunDialog(QDialog):
    simulation_done = Signal(bool, str)
    simulation_termination_request = Signal()

    def __init__(self, config_file, run_model, simulation_arguments, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowFlags(Qt.Window)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModal)
        self.setWindowTitle("Simulations - {}".format(config_file))

        self._snapshot_model = SnapshotModel(self)
        self._run_model = run_model

        self._isDetailedDialog = False
        self._minimum_width = 1200

        ert = None
        if isinstance(run_model, BaseRunModel):
            ert = run_model.ert()

        self._simulations_argments = simulation_arguments

        self._ticker = QTimer(self)
        self._ticker.timeout.connect(self._on_ticker)

        progress_proxy_model = ProgressProxyModel(self._snapshot_model, parent=self)

        self._total_progress_label = QLabel(
            _TOTAL_PROGRESS_TEMPLATE.format(
                total_progress=0, phase_name=run_model.getPhaseName()
            ),
            self,
        )

        self._total_progress_bar = QProgressBar(self)
        self._total_progress_bar.setRange(0, 100)
        self._total_progress_bar.setTextVisible(False)

        self._iteration_progress_label = QLabel(self)

        self._progress_view = ProgressView(self)
        self._progress_view.setModel(progress_proxy_model)
        self._progress_view.setIndeterminate(True)

        legend_view = LegendView(self)
        legend_view.setModel(progress_proxy_model)

        self._tab_widget = QTabWidget(self)
        self._tab_widget.currentChanged.connect(self._current_tab_changed)
        self._snapshot_model.rowsInserted.connect(self.on_new_iteration)

        self._job_label = QLabel(self)

        self._job_model = JobListProxyModel(self, 0, 0, 0, 0)
        self._job_model.setSourceModel(self._snapshot_model)

        self._job_view = QTableView(self)
        self._job_view.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)
        self._job_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._job_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self._job_view.clicked.connect(self._job_clicked)
        self._open_files = {}
        self._job_view.setModel(self._job_model)

        self.running_time = QLabel("")

        self.plot_tool = PlotTool(config_file)
        self.plot_tool.setParent(self)
        self.plot_button = QPushButton(self.plot_tool.getName())
        self.plot_button.clicked.connect(self.plot_tool.trigger)
        self.plot_button.setEnabled(ert is not None)

        self.kill_button = QPushButton("Kill Simulations")
        self.done_button = QPushButton("Done")
        self.done_button.setHidden(True)
        self.restart_button = QPushButton("Restart")
        self.restart_button.setHidden(True)
        self.show_details_button = QPushButton("Show Details")
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

        layout = QVBoxLayout()
        layout.addWidget(self._total_progress_label)
        layout.addWidget(self._total_progress_bar)
        layout.addWidget(self._iteration_progress_label)
        layout.addWidget(self._progress_view)
        layout.addWidget(legend_view)
        layout.addWidget(self._tab_widget)
        layout.addWidget(self._job_label)
        layout.addWidget(self._job_view)
        layout.addWidget(button_widget_container)

        self.setLayout(layout)

        self.kill_button.clicked.connect(self.killJobs)
        self.done_button.clicked.connect(self.accept)
        self.restart_button.clicked.connect(self.restart_failed_realizations)
        self.show_details_button.clicked.connect(self.toggle_detailed_progress)
        self.simulation_done.connect(self._on_simulation_done)

        self.setMinimumWidth(self._minimum_width)
        self._setSimpleDialog()

    def _current_tab_changed(self, index: int):
        # Clear the selection in the other tabs
        for i in range(0, self._tab_widget.count()):
            if i != index:
                self._tab_widget.widget(i).clearSelection()

    def _setSimpleDialog(self) -> None:
        self._isDetailedDialog = False
        self._tab_widget.setVisible(False)
        self._job_label.setVisible(False)
        self._job_view.setVisible(False)
        self.show_details_button.setText("Show Details")

    def _setDetailedDialog(self) -> None:
        self._isDetailedDialog = True
        self._tab_widget.setVisible(True)
        self._job_label.setVisible(True)
        self._job_view.setVisible(True)
        self.show_details_button.setText("Hide Details")

    @Slot(QModelIndex, int, int)
    def on_new_iteration(self, parent: QModelIndex, start: int, end: int) -> None:
        if not parent.isValid():
            index = self._snapshot_model.index(start, 0, parent)
            iter_row = start
            self._iteration_progress_label.setText(
                f"Progress for iteration {index.internalPointer().id}"
            )

            widget = RealizationWidget(iter_row)
            widget.setSnapshotModel(self._snapshot_model)
            widget.currentChanged.connect(self._select_real)

            self._tab_widget.addTab(
                widget, f"Realizations for iteration {index.internalPointer().id}"
            )

    @Slot(QModelIndex)
    def _job_clicked(self, index):
        if not index.isValid():
            return
        selected_file = index.data(FileRole)

        if selected_file and selected_file not in self._open_files:
            job_name = index.siblingAtColumn(0).data()
            viewer = FileDialog(
                selected_file,
                job_name,
                index.row(),
                index.model().get_real(),
                index.model().get_iter(),
                self,
            )
            self._open_files[selected_file] = viewer
            viewer.finished.connect(lambda _, f=selected_file: self._open_files.pop(f))

        elif selected_file in self._open_files:
            self._open_files[selected_file].raise_()

    @Slot(QModelIndex)
    def _select_real(self, index):
        step = 0
        stage = 0
        real = index.row()
        iter_ = index.model().get_iter()
        self._job_model.set_step(iter_, real, stage, step)
        self._job_label.setText(
            f"Realization id {index.data(RealIens)} in iteration {iter_}"
        )

        self._job_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def reject(self):
        return

    def closeEvent(self, QCloseEvent):
        if self._run_model.isFinished():
            self.simulation_done.emit(
                self._run_model.hasRunFailed(), self._run_model.getFailMessage()
            )
        else:
            # Kill jobs if dialog is closed
            if self.killJobs() != QMessageBox.Yes:
                QCloseEvent.ignore()

    def startSimulation(self):
        self._run_model.reset()
        self._snapshot_model.reset()
        self._tab_widget.clear()

        def run():
            asyncio.set_event_loop(asyncio.new_event_loop())
            self._run_model.startSimulations(self._simulations_argments)

        simulation_thread = Thread(name="ert_gui_simulation_thread")
        simulation_thread.setDaemon(True)
        simulation_thread.run = run
        simulation_thread.start()

        self._ticker.start(1000)

        tracker = create_tracker(
            self._run_model,
            num_realizations=self._simulations_argments["active_realizations"].count(),
            ee_config=self._simulations_argments.get("ee_config", None),
        )

        worker = TrackerWorker(tracker)
        worker_thread = QThread()
        worker.done.connect(worker_thread.quit)
        worker.consumed_event.connect(self._on_tracker_event)
        worker.moveToThread(worker_thread)
        self.simulation_done.connect(worker.stop)
        self._worker = worker
        self._worker_thread = worker_thread
        worker_thread.started.connect(worker.consume_and_emit)
        self._worker_thread.start()

    def killJobs(self):

        msg = "Are you sure you want to kill the currently running simulations?"
        if self._run_model.getQueueStatus().get(JobStatusType.JOB_QUEUE_UNKNOWN, 0) > 0:
            msg += "\n\nKilling a simulation with unknown status will not kill the realizations already submitted!"
        kill_job = QMessageBox.question(
            self, "Kill simulations?", msg, QMessageBox.Yes | QMessageBox.No
        )

        if kill_job == QMessageBox.Yes:
            # Normally this slot would be invoked by the signal/slot system,
            # but the worker is busy tracking the evaluation.
            self._worker.request_termination()
            self.reject()
        return kill_job

    @Slot(bool, str)
    def _on_simulation_done(self, failed, failed_msg):
        self.processing_animation.hide()
        self.kill_button.setHidden(True)
        self.done_button.setHidden(False)
        self.restart_button.setVisible(self.has_failed_realizations())
        self.restart_button.setEnabled(self._run_model.support_restart)
        self._total_progress_bar.setValue(100)
        self._total_progress_label.setText(
            _TOTAL_PROGRESS_TEMPLATE.format(
                total_progress=100, phase_name=self._run_model.getPhaseName()
            )
        )

        if failed:
            QMessageBox.critical(
                self,
                "Simulations failed!",
                f"The simulation failed with the following error:\n\n{failed_msg}",
            )

    @Slot()
    def _on_ticker(self):
        runtime = self._run_model.get_runtime()
        self.running_time.setText(format_running_time(runtime))

    @Slot(object)
    def _on_tracker_event(self, event):
        if isinstance(event, EndEvent):
            self.simulation_done.emit(event.failed, event.failed_msg)
            self._worker.stop()
            self._ticker.stop()

        elif isinstance(event, FullSnapshotEvent):
            if event.snapshot is not None:
                self._snapshot_model._add_snapshot(event.snapshot, event.iteration)
            self._progress_view.setIndeterminate(event.indeterminate)
            progress = int(event.progress * 100)
            self._total_progress_bar.setValue(progress)
            self._total_progress_label.setText(
                _TOTAL_PROGRESS_TEMPLATE.format(
                    total_progress=progress, phase_name=event.phase_name
                )
            )

        elif isinstance(event, SnapshotUpdateEvent):
            if event.partial_snapshot is not None:
                self._snapshot_model._add_partial_snapshot(
                    event.partial_snapshot, event.iteration
                )
            self._progress_view.setIndeterminate(event.indeterminate)
            progress = int(event.progress * 100)
            self._total_progress_bar.setValue(progress)
            self._total_progress_label.setText(
                _TOTAL_PROGRESS_TEMPLATE.format(
                    total_progress=progress, phase_name=event.phase_name
                )
            )

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
        inverted_mask = BoolVector(default_value=False)
        for (index, successful) in enumerate(completed):
            inverted_mask[index] = initial[index] and not successful
        return inverted_mask

    def restart_failed_realizations(self):

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setText(
            "Note that workflows will only be executed on the restarted realizations and that this might have unexpected consequences."
        )
        msg.setWindowTitle("Restart Failed Realizations")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        result = msg.exec_()

        if result == QMessageBox.Ok:
            self.restart_button.setVisible(False)
            self.kill_button.setVisible(True)
            self.done_button.setVisible(False)
            active_realizations = self.create_mask_from_failed_realizations()
            self._simulations_argments["active_realizations"] = active_realizations
            self._simulations_argments[
                "prev_successful_realizations"
            ] = self._simulations_argments.get("prev_successful_realizations", 0)
            self._simulations_argments[
                "prev_successful_realizations"
            ] += self.count_successful_realizations()
            self.startSimulation()

    @Slot()
    def toggle_detailed_progress(self):
        if self._isDetailedDialog:
            self._setSimpleDialog()
        else:
            self._setDetailedDialog()

        self.adjustSize()
