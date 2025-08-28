from __future__ import annotations

import logging
from pathlib import Path
from queue import SimpleQueue
from typing import cast

import humanize
from PyQt6.QtCore import QModelIndex, QSize, Qt, QThread, QTimer
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtGui import QHideEvent, QMouseEvent, QMovie, QTextCursor, QTextOption
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ert.config import ErrorInfo, QueueSystem, WarningInfo
from ert.ensemble_evaluator import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
    WarningEvent,
)
from ert.ensemble_evaluator import identifiers as ids
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.model.fm_step_list import FMStepListProxyModel
from ert.gui.model.node import IterNode
from ert.gui.model.real_list import RealListModel
from ert.gui.model.snapshot import (
    FM_STEP_COLUMNS,
    FileRole,
    IterNum,
    RealIens,
    SnapshotModel,
)
from ert.gui.suggestor import Suggestor
from ert.gui.tools.file import FileDialog
from ert.run_models import (
    RunModelAPI,
    RunModelStatusEvent,
    RunModelTimeEvent,
    RunModelUpdateBeginEvent,
    RunModelUpdateEndEvent,
    StatusEvents,
)
from ert.run_models.event import (
    EverestBatchResultEvent,
    RunModelDataEvent,
    RunModelErrorEvent,
)
from ert.shared.status.utils import (
    byte_with_unit,
    file_has_content,
    get_mount_directory,
)

from .queue_emitter import QueueEmitter
from .view import DiskSpaceWidget, ProgressWidget, RealizationWidget, UpdateWidget
from .view.disk_space_widget import MountType

_TOTAL_PROGRESS_TEMPLATE = "Total progress {total_progress}% — {iteration_label}"
_EVEREST_TOTAL_PROGRESS_TEMPLATE = "Batch {iteration} progress: {total_progress}%"

logger = logging.getLogger(__name__)


def _batch_type_text(batch_id: int, batch_types: set[str]) -> str:
    """
    >>> _batch_type_text(1, {"FunctionResult"})
    'Batch 1: fn'
    >>> _batch_type_text(2, {"GradientResult"})
    'Batch 2: ∇'
    >>> _batch_type_text(3, {"FunctionResult", "GradientResult"})
    'Batch 3: fn+∇'
    """
    type_text = ""
    if batch_types == {"FunctionResult", "GradientResult"}:
        type_text = "fn+∇"
    elif batch_types == {"FunctionResult"}:
        type_text = "fn"
    elif batch_types == {"GradientResult"}:
        type_text = "∇"

    return f"Batch {batch_id}: {type_text}"


class FMStepOverview(QTableView):
    def __init__(self, snapshot_model: SnapshotModel, parent: QWidget | None) -> None:
        super().__init__(parent)

        self._fm_step_model = FMStepListProxyModel(self, 0, 0)
        self._fm_step_model.setSourceModel(snapshot_model)

        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerItem)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        self.clicked.connect(self._fm_step_clicked)
        self.setModel(self._fm_step_model)

        horizontal_header = self.horizontalHeader()
        assert horizontal_header is not None

        horizontal_header.resizeSections(QHeaderView.ResizeMode.ResizeToContents)
        for section in range(horizontal_header.count()):
            if horizontal_header.sectionSize(section) < 135:
                horizontal_header.resizeSection(section, 135)

            # Only last section should be stretch
            horizontal_header.setSectionResizeMode(
                section,
                QHeaderView.ResizeMode.Stretch
                if section == horizontal_header.count() - 1
                else QHeaderView.ResizeMode.Interactive,
            )

        vertical_header = self.verticalHeader()
        assert vertical_header is not None
        vertical_header.setMinimumWidth(20)
        self.setMinimumHeight(140)
        self.setMouseTracking(True)

    @Slot(int, int)
    def set_realization(self, iter_: int, real: int) -> None:
        self._fm_step_model.set_real(iter_, real)

    @Slot(QModelIndex)
    def _fm_step_clicked(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        selected_file = index.data(FileRole)
        file_dialog = self.findChild(QDialog, name=selected_file)
        if file_dialog and file_dialog.isVisible():
            file_dialog.raise_()
        elif selected_file and file_has_content(selected_file):
            fm_step_name = index.siblingAtColumn(0).data()
            FileDialog(
                selected_file,
                fm_step_name,
                index.row(),
                index.data(RealIens),
                index.data(IterNum),
                self,
            )
        elif FM_STEP_COLUMNS[index.column()] == ids.ERROR and index.data():
            error_dialog = QDialog(self)
            error_dialog.setWindowTitle("Error information")
            layout = QVBoxLayout(error_dialog)

            error_textedit = QPlainTextEdit()
            error_textedit.setReadOnly(True)
            error_textedit.setWordWrapMode(QTextOption.WrapMode.NoWrap)
            error_textedit.appendPlainText(index.data())
            layout.addWidget(error_textedit)

            dialog_button = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
            dialog_button.accepted.connect(error_dialog.accept)
            layout.addWidget(dialog_button)
            error_dialog.resize(700, 300)
            error_textedit.moveCursor(QTextCursor.MoveOperation.Start)
            error_dialog.exec()

    def mouseMoveEvent(self, e: QMouseEvent | None) -> None:
        if e:
            index = self.indexAt(e.pos())
            if index.isValid():
                data_name = FM_STEP_COLUMNS[index.column()]
                if data_name in {ids.STDOUT, ids.STDERR} and file_has_content(
                    index.data(FileRole)
                ):
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)

        return super().mouseMoveEvent(e)


class RunDialog(QFrame):
    simulation_done = Signal(bool, str)
    progress_update_event = Signal(dict, int)
    rerun_failed_realizations_experiment = Signal()
    _RUN_TIME_POLL_RATE = 1000

    def __init__(
        self,
        title: str,
        run_model_api: RunModelAPI,
        event_queue: SimpleQueue[StatusEvents],
        notifier: ErtNotifier,
        parent: QWidget | None = None,
        output_path: Path | None = None,
        is_everest: bool | None = False,
        run_path: Path | None = None,
        storage_path: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.run_path = run_path or Path()
        self.storage_path = storage_path or Path()
        self.output_path = output_path
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowTitle(title)

        self._run_model_api = run_model_api
        self._snapshot_model = SnapshotModel(self)
        self._event_queue = event_queue
        self._notifier = notifier
        self.fail_msg_box: Suggestor | None = None
        self.post_simulation_warnings: list[str] = []

        self._ticker = QTimer(self)
        self._ticker.timeout.connect(self._on_ticker)

        self._is_everest = is_everest

        if is_everest:
            self._batch_result_types: list[set[str]] = []

        self._total_progress_label = QLabel(
            _TOTAL_PROGRESS_TEMPLATE.format(
                total_progress=0, iteration_label="Starting..."
            ),
            self,
        )

        self._total_progress_bar = QProgressBar(self)
        self._total_progress_bar.setRange(0, 100)
        self._total_progress_bar.setTextVisible(False)
        self._total_progress_bar_calculated_value = -1

        self._iteration_progress_label = QLabel(self)
        self._progress_widget = ProgressWidget()

        self._tab_widget = QTabWidget(self)
        self._tab_widget.setMinimumHeight(250)
        self._tab_widget.currentChanged.connect(self._current_tab_changed)
        self._snapshot_model.rowsInserted.connect(self.on_snapshot_new_iteration)

        self._fm_step_label = QLabel(self)
        self._fm_step_label.setObjectName("fm_step_label")
        self._fm_step_overview = FMStepOverview(self._snapshot_model, self)

        self.running_time = QLabel("Running time:\n -")
        self.running_time.setMinimumWidth(150)
        self.queue_system = QLabel("")
        self.queue_system.setMinimumWidth(150)
        self.memory_usage = QLabel("Maximal realization memory usage: \n -")
        self.memory_usage.setMinimumWidth(250)

        self.kill_button = QPushButton("Terminate experiment")
        self.rerun_button = QPushButton("Rerun failed simulations")
        self.rerun_button.setEnabled(False)

        size = 20
        spin_movie = QMovie("img:loading.gif")
        spin_movie.setSpeed(60)
        spin_movie.setScaledSize(QSize(size, size))
        spin_movie.start()

        self.processing_animation = QLabel()
        self.processing_animation.setFixedSize(QSize(size, size))
        self.processing_animation.setMovie(spin_movie)
        self.processing_stopped = QLabel()
        self.processing_stopped.setFixedSize(QSize(size, size))
        self.processing_stopped.setVisible(False)

        footer_layout = QHBoxLayout()
        footer_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        running_time_layout = QHBoxLayout()
        running_time_layout.addWidget(self.processing_animation)
        running_time_layout.addWidget(self.processing_stopped)
        running_time_layout.addWidget(self.running_time)
        footer_layout.addLayout(running_time_layout)

        footer_layout.addWidget(self.queue_system)
        footer_layout.addWidget(self.memory_usage)

        self.disk_space_runpath = DiskSpaceWidget(
            get_mount_directory(self.run_path),
            MountType.RUNPATH,
        )
        footer_layout.addWidget(self.disk_space_runpath)
        self.disk_widgets = [self.disk_space_runpath]

        self.disk_space_storage = DiskSpaceWidget(
            get_mount_directory(self.storage_path),
            MountType.STORAGE,
        )
        footer_layout.addWidget(self.disk_space_storage)
        self.disk_widgets.append(self.disk_space_storage)

        footer_layout.addStretch(1000)
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.kill_button)
        button_layout.addWidget(self.rerun_button)
        footer_layout.addLayout(button_layout)

        footer_widget_container = QWidget()
        footer_widget_container.setLayout(footer_layout)

        layout = QVBoxLayout()
        layout.addWidget(self._total_progress_label)
        layout.addWidget(self._total_progress_bar)
        layout.addWidget(self._iteration_progress_label)
        layout.addWidget(self._progress_widget)

        adjustable_splitter_layout = QSplitter()
        adjustable_splitter_layout.setOrientation(Qt.Orientation.Vertical)
        adjustable_splitter_layout.addWidget(self._tab_widget)

        adjustable_splitter_layout.setStyleSheet("""
            QSplitter::handle {
                image: url(img:drag_handle.svg);
                height: 13px;
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                   stop: 0.1 #10FFFFFF, stop: 0.5 #D3D3D3, stop: 0.9 #10FFFFFF);
            }
         """)

        self.fm_step_frame = QFrame(self)
        fm_step_frame_layout = QVBoxLayout(self.fm_step_frame)
        fm_step_frame_layout.setContentsMargins(0, 0, 0, 0)
        fm_step_frame_layout.addWidget(self._fm_step_label)
        fm_step_frame_layout.addWidget(self._fm_step_overview)

        adjustable_splitter_layout.addWidget(self.fm_step_frame)
        layout.addWidget(adjustable_splitter_layout)
        layout.addWidget(footer_widget_container)

        self.setLayout(layout)

        self.kill_button.clicked.connect(self.killJobs)
        self.rerun_button.clicked.connect(self.rerun_failed_realizations)
        self.simulation_done.connect(self._on_simulation_done)

        self.setMinimumSize(1200, 600)
        self._is_rerunning_failed_realizations = False
        self.flag_simulation_done = False

        self._latest_iteration = 0

    def is_simulation_done(self) -> bool:
        return self.flag_simulation_done

    def _current_tab_changed(self, index: int) -> None:
        widget = self._tab_widget.widget(index)
        if isinstance(widget, RealizationWidget):
            widget.refresh_current_selection()

        self.fm_step_frame.setHidden(isinstance(widget, UpdateWidget))

    @Slot(QModelIndex, int, int)
    def on_snapshot_new_iteration(
        self, parent: QModelIndex, start: int, end: int
    ) -> None:
        if not parent.isValid():
            index = self._snapshot_model.index(start, 0, parent)
            iteration = int(cast(IterNode, index.internalPointer()).id_)
            self._latest_iteration = iteration
            iter_row = start
            self._iteration_progress_label.setText(
                f"Progress for iteration {iteration}"
                if not self._is_everest
                else f"Progress for batch {iteration}"
            )

            widget = RealizationWidget(iter_row)
            widget.setSnapshotModel(self._snapshot_model)
            widget.itemClicked.connect(self._select_real)
            widget.setProperty("identifier", f"tab-iter-{iteration}")
            self._select_real(widget._real_list_model.index(0, 0))
            tab_index = self._tab_widget.addTab(
                widget,
                f"Realizations for iteration {iteration}"
                if not self._is_everest
                else f"Batch {iteration}...",
            )
            if self._tab_widget.currentIndex() == self._tab_widget.count() - 2:
                self._tab_widget.setCurrentIndex(tab_index)

            if self._is_everest:
                self._batch_result_types.append(set())

    @Slot(QModelIndex)
    def _select_real(self, index: QModelIndex) -> None:
        if index.isValid():
            real = index.row()
            iter_ = cast(RealListModel, index.model()).get_iter()
            exec_hosts = None

            iter_node = self._snapshot_model.root.children.get(str(iter_), None)
            if iter_node:
                real_node = iter_node.children.get(str(real), None)
                if real_node:
                    exec_hosts = real_node.data.exec_hosts

            self._fm_step_overview.set_realization(iter_, real)

            if not self._is_everest:
                text = (
                    f"Realization id {index.data(RealIens)} in "
                    f"iteration {index.data(IterNum)}"
                )
            else:
                text = (
                    f"Simulation {index.data(RealIens)} in batch {index.data(IterNum)}"
                )

            if exec_hosts and exec_hosts != "-":
                text += f", assigned to host: {exec_hosts}"
            self._fm_step_label.setText(text)

    def setup_event_monitoring(self, rerun_failed_realizations: bool = False) -> None:
        self.flag_simulation_done = False
        if rerun_failed_realizations is False:
            self._snapshot_model.reset()
            self._tab_widget.clear()

        self._worker_thread = QThread(parent=self)

        self._worker = QueueEmitter(self._event_queue)
        self._worker.done.connect(self._worker_thread.quit)
        self._worker.new_event.connect(self._on_event)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.consume_and_emit)
        self.simulation_done.connect(self._worker.stop)
        self.destroyed.connect(lambda: _stop_worker(self._worker_thread, self._worker))
        self._worker_thread.start()
        self._ticker.start(self._RUN_TIME_POLL_RATE)

    def killJobs(self) -> QMessageBox.StandardButton:
        msg = "Are you sure you want to terminate the currently running experiment?"
        kill_job = QMessageBox.question(
            self,
            "Terminate experiment",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if kill_job == QMessageBox.StandardButton.Yes:
            # Normally this slot would be invoked by the signal/slot system,
            # but the worker is busy tracking the evaluation.
            self._run_model_api.cancel()
        return kill_job

    @Slot(bool, str)
    def _on_simulation_done(self, failed: bool, msg: str) -> None:
        self.processing_animation.setVisible(False)
        self.processing_stopped.setVisible(True)
        self.kill_button.setEnabled(False)
        self.rerun_button.setEnabled(
            self._run_model_api.has_failed_realizations()
            and self._run_model_api.supports_rerunning_failed_realizations
        )
        self._notifier.set_is_simulation_running(False)
        self.flag_simulation_done = True

        if failed:
            self.update_total_progress(1.0, "Failed")
            self._progress_widget.set_all_failed()
        else:
            self.update_total_progress(1.0, "Experiment completed.")

        if failed or self.post_simulation_warnings:
            self.fail_msg_box = Suggestor(
                errors=[ErrorInfo(msg)] if failed else [],
                warnings=[WarningInfo(msg) for msg in self.post_simulation_warnings],
                deprecations=[],
                continue_action=None,
                widget_info=(
                    f"<p style='font-size: 28px;' > ERT experiment "
                    f"{'failed' if failed else 'succeeded'}!</p>"
                    f"<p style='font-size: 16px;'>"
                    f"These {'errors' if failed else 'warnings'} were detected</p>"
                ),
                parent=self,
            )
            self.fail_msg_box.show()

        if self.post_simulation_warnings:
            logger.info(
                f"Simulation finished with "
                f"{len(self.post_simulation_warnings)} PostSimulationWarnings"
            )

    @Slot()
    def _on_ticker(self) -> None:
        runtime = self._run_model_api.get_runtime()
        running_time = f"Running time: {humanize.precisedelta(runtime)}"
        self.running_time.setText(running_time[0:14] + "\n" + running_time[14:])

        maximum_memory_usage = self._snapshot_model.root.max_memory_usage

        for disk_widget in self.disk_widgets:
            disk_widget.update_status()

        if maximum_memory_usage:
            self.memory_usage.setText(
                "Maximal realization memory usage: \n"
                f"{byte_with_unit(maximum_memory_usage)}"
            )

    @Slot(object)
    def _on_event(self, event: object) -> None:
        model = self._snapshot_model
        match event:
            case EndEvent(failed=failed, msg=msg):
                self.simulation_done.emit(failed, msg)
                self._ticker.stop()
            case WarningEvent(msg=msg):
                self.post_simulation_warnings.append(msg)
            case FullSnapshotEvent(
                status_count=status_count, realization_count=realization_count
            ):
                if event.snapshot is not None:
                    if self._is_rerunning_failed_realizations:
                        model._update_snapshot(event.snapshot, str(event.iteration))
                    else:
                        model._add_snapshot(event.snapshot, str(event.iteration))
                self.update_total_progress(
                    event.progress, event.iteration_label, event.iteration
                )
                self._progress_widget.update_progress(status_count, realization_count)
                self.progress_update_event.emit(status_count, realization_count)
            case SnapshotUpdateEvent(
                status_count=status_count, realization_count=realization_count
            ):
                if event.snapshot is not None:
                    model._update_snapshot(event.snapshot, str(event.iteration))
                self._progress_widget.update_progress(status_count, realization_count)
                self.update_total_progress(
                    event.progress, event.iteration_label, event.iteration
                )
                self.progress_update_event.emit(status_count, realization_count)
            case RunModelUpdateBeginEvent(iteration=iteration):
                widget = UpdateWidget(iteration)
                tab_index = self._tab_widget.addTab(widget, f"Update {iteration}")
                if self._tab_widget.currentIndex() == self._tab_widget.count() - 2:
                    self._tab_widget.setCurrentIndex(tab_index)
                widget.begin(event)
            case RunModelUpdateEndEvent():
                self._progress_widget.stop_waiting_progress_bar()
                self._get_update_widget(event.iteration).end(event)
                event.write_as_csv(self.output_path)
            case RunModelStatusEvent() | RunModelTimeEvent():
                self._get_update_widget(event.iteration).update_status(event)
            case RunModelDataEvent():
                self._get_update_widget(event.iteration).add_table(event)
                event.write_as_csv(self.output_path)
            case RunModelErrorEvent():
                self._get_update_widget(event.iteration).error(event)
                event.write_as_csv(self.output_path)
            case EverestBatchResultEvent():
                batch_types = self._batch_result_types[event.batch]
                batch_types.add(event.result_type)

                self._tab_widget.setTabText(
                    event.batch, _batch_type_text(event.batch, batch_types)
                )

    def _get_update_widget(self, iteration: int) -> UpdateWidget:
        for i in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(i)
            if isinstance(widget, UpdateWidget) and widget.iteration == iteration:
                return widget
        raise ValueError("Could not find UpdateWidget")

    def update_total_progress(
        self, progress_value: float, iteration_label: str, iteration: int | None = None
    ) -> None:
        if iteration is None:
            iteration = self._latest_iteration

        progress = int(progress_value * 100)

        if (
            progress < 0 or progress > 100
        ) and progress != self._total_progress_bar_calculated_value:
            logger.warning(f"Total progress bar exceeds [0-100] range: {progress}")
            self._total_progress_bar_calculated_value = progress

        self._total_progress_bar.setValue(progress)

        if self._is_everest:
            self._total_progress_label.setText(
                _EVEREST_TOTAL_PROGRESS_TEMPLATE.format(
                    total_progress=progress, iteration=iteration
                )
            )
        else:
            self._total_progress_label.setText(
                _TOTAL_PROGRESS_TEMPLATE.format(
                    total_progress=progress, iteration_label=iteration_label
                )
            )

    def rerun_failed_realizations(self) -> None:
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(
            "Note that workflows will only be executed on the restarted "
            "realizations and that this might have unexpected consequences."
        )
        msg.setWindowTitle("Restart failed realizations")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        msg.setObjectName("restart_prompt")
        result = msg.exec()

        if result == QMessageBox.StandardButton.Ok:
            self.rerun_button.setEnabled(False)
            self.kill_button.setEnabled(True)
            self._is_rerunning_failed_realizations = True
            self.rerun_failed_realizations_experiment.emit()

    def set_queue_system_name(self, queue_system: QueueSystem) -> None:
        match queue_system:
            case QueueSystem.LSF:
                formatted_queue_system = "LSF"
            case QueueSystem.LOCAL:
                formatted_queue_system = "Local"
            case QueueSystem.TORQUE:
                formatted_queue_system = "Torque/OpenPBS"
            case QueueSystem.SLURM:
                formatted_queue_system = "Slurm"
        self.queue_system.setText(f"Queue system:\n{formatted_queue_system}")

    def hideEvent(self, event: QHideEvent | None) -> None:
        for file_dialog in self.findChildren(FileDialog):
            file_dialog.close()


# Cannot use a non-static method here as
# it is called when the object is destroyed
# https://stackoverflow.com/questions/16842955
def _stop_worker(worker_thread: QThread, worker: QueueEmitter) -> None:
    if worker_thread.isRunning():
        worker.stop()
        worker_thread.wait(3000)
    if worker_thread.isRunning():
        worker_thread.quit()
        worker_thread.wait(3000)
    if worker_thread.isRunning():
        worker_thread.terminate()
        worker_thread.wait(3000)
