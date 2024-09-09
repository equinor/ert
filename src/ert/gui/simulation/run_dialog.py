from __future__ import annotations

import logging
from pathlib import Path
from queue import SimpleQueue
from typing import Optional

from qtpy.QtCore import QModelIndex, QSize, Qt, QThread, QTimer, Signal, Slot
from qtpy.QtGui import (
    QCloseEvent,
    QKeyEvent,
    QMouseEvent,
    QMovie,
    QTextCursor,
    QTextOption,
)
from qtpy.QtWidgets import (
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

from _ert.threading import ErtThread
from ert.config import QueueSystem
from ert.ensemble_evaluator import (
    EndEvent,
    EvaluatorServerConfig,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.ensemble_evaluator import identifiers as ids
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.message_box import ErtMessageBox
from ert.gui.model.fm_step_list import FMStepListProxyModel
from ert.gui.model.snapshot import (
    FM_STEP_COLUMNS,
    FileRole,
    IterNum,
    RealIens,
    SnapshotModel,
)
from ert.gui.tools.file import FileDialog
from ert.gui.tools.plot.plot_tool import PlotTool
from ert.run_models import (
    BaseRunModel,
    RunModelStatusEvent,
    RunModelTimeEvent,
    RunModelUpdateBeginEvent,
    RunModelUpdateEndEvent,
    StatusEvents,
)
from ert.run_models.event import RunModelDataEvent, RunModelErrorEvent
from ert.shared.status.utils import (
    byte_with_unit,
    file_has_content,
    format_running_time,
)

from ..find_ert_info import find_ert_info
from .queue_emitter import QueueEmitter
from .view import ProgressWidget, RealizationWidget, UpdateWidget

_TOTAL_PROGRESS_TEMPLATE = "Total progress {total_progress}% — {iteration_label}"


class FMStepOverview(QTableView):
    def __init__(self, snapshot_model: SnapshotModel, parent: QWidget | None) -> None:
        super().__init__(parent)

        self._fm_step_model = FMStepListProxyModel(self, 0, 0)
        self._fm_step_model.setSourceModel(snapshot_model)

        self.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)

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
                QHeaderView.Stretch
                if section == horizontal_header.count() - 1
                else QHeaderView.Interactive,
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
            error_textedit.setWordWrapMode(QTextOption.NoWrap)
            error_textedit.appendPlainText(index.data())
            layout.addWidget(error_textedit)

            dialog_button = QDialogButtonBox(QDialogButtonBox.Ok)
            dialog_button.accepted.connect(error_dialog.accept)
            layout.addWidget(dialog_button)
            error_dialog.resize(700, 300)
            error_textedit.moveCursor(QTextCursor.Start)
            error_dialog.exec_()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event:
            index = self.indexAt(event.pos())
            if index.isValid():
                data_name = FM_STEP_COLUMNS[index.column()]
                if data_name in [ids.STDOUT, ids.STDERR] and file_has_content(
                    index.data(FileRole)
                ):
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)

        return super().mouseMoveEvent(event)


class RunDialog(QDialog):
    simulation_done = Signal(bool, str)
    produce_clipboard_debug_info = Signal()
    on_run_model_event = Signal(object)
    _RUN_TIME_POLL_RATE = 1000

    def __init__(
        self,
        config_file: str,
        run_model: BaseRunModel,
        event_queue: SimpleQueue[StatusEvents],
        notifier: ErtNotifier,
        parent: Optional[QWidget] = None,
        output_path: Optional[Path] = None,
    ):
        QDialog.__init__(self, parent)
        self.output_path = output_path
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # type: ignore
        self.setWindowTitle(f"Experiment - {config_file} {find_ert_info()}")

        self._snapshot_model = SnapshotModel(self)
        self._run_model = run_model
        self._event_queue = event_queue
        self._notifier = notifier

        self._minimum_width = 1200
        self._minimum_height = 800

        self._ticker = QTimer(self)
        self._ticker.timeout.connect(self._on_ticker)

        self._total_progress_label = QLabel(
            _TOTAL_PROGRESS_TEMPLATE.format(
                total_progress=0, iteration_label="Starting..."
            ),
            self,
        )

        self._total_progress_bar = QProgressBar(self)
        self._total_progress_bar.setRange(0, 100)
        self._total_progress_bar.setTextVisible(False)

        self._iteration_progress_label = QLabel(self)
        self._progress_widget = ProgressWidget()

        self._tab_widget = QTabWidget(self)
        self._tab_widget.setMinimumHeight(250)
        self._tab_widget.currentChanged.connect(self._current_tab_changed)
        self._snapshot_model.rowsInserted.connect(self.on_snapshot_new_iteration)

        self._fm_step_label = QLabel(self)
        self._fm_step_overview = FMStepOverview(self._snapshot_model, self)

        self.running_time = QLabel("")
        self.memory_usage = QLabel("")

        self.plot_tool = PlotTool(config_file, self.parent())  # type: ignore
        self.plot_button = QPushButton(self.plot_tool.getName())
        self.plot_button.clicked.connect(self.plot_tool.trigger)
        self.plot_button.setEnabled(True)

        self.kill_button = QPushButton("Terminate experiment")
        self.done_button = QPushButton("Done")
        self.done_button.setHidden(True)
        self.restart_button = QPushButton("Restart")
        self.restart_button.setHidden(True)
        self.copy_debug_info_button = QPushButton("Debug Info")
        self.copy_debug_info_button.setToolTip("Copies useful information to clipboard")
        self.copy_debug_info_button.clicked.connect(self.produce_clipboard_debug_info)
        self.copy_debug_info_button.setObjectName("copy_debug_info_button")

        size = 20
        spin_movie = QMovie("img:loading.gif")
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
        button_layout.addWidget(self.memory_usage)
        button_layout.addStretch()
        button_layout.addWidget(self.copy_debug_info_button)
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
        layout.addWidget(button_widget_container)

        self.setLayout(layout)

        self.kill_button.clicked.connect(self.killJobs)  # type: ignore
        self.done_button.clicked.connect(self.accept)
        self.restart_button.clicked.connect(self.restart_failed_realizations)
        self.simulation_done.connect(self._on_simulation_done)

        self.setMinimumSize(self._minimum_width, self._minimum_height)
        self.finished.connect(self._on_finished)

        self.on_run_model_event.connect(self._on_event)

    def _current_tab_changed(self, index: int) -> None:
        widget = self._tab_widget.widget(index)
        self.fm_step_frame.setHidden(isinstance(widget, UpdateWidget))

    @Slot(QModelIndex, int, int)
    def on_snapshot_new_iteration(
        self, parent: QModelIndex, start: int, end: int
    ) -> None:
        if not parent.isValid():
            index = self._snapshot_model.index(start, 0, parent)
            iter_row = start
            self._iteration_progress_label.setText(
                f"Progress for iteration {index.internalPointer().id_}"
            )

            widget = RealizationWidget(iter_row)
            widget.setSnapshotModel(self._snapshot_model)
            widget.itemClicked.connect(self._select_real)

            tab_index = self._tab_widget.addTab(
                widget, f"Realizations for iteration {index.internalPointer().id_}"
            )
            if self._tab_widget.currentIndex() == self._tab_widget.count() - 2:
                self._tab_widget.setCurrentIndex(tab_index)

    @Slot(QModelIndex)
    def _select_real(self, index: QModelIndex) -> None:
        real = index.row()
        iter_ = index.model().get_iter()  # type: ignore
        self._fm_step_overview.set_realization(iter_, real)
        self._fm_step_label.setText(
            f"Realization id {index.data(RealIens)} in iteration {index.data(IterNum)}"
        )

    def closeEvent(self, a0: Optional[QCloseEvent]) -> None:
        if not self._notifier.is_simulation_running:
            self.accept()
        elif self.killJobs() != QMessageBox.Yes and a0 is not None:
            a0.ignore()

    def run_experiment(self, restart: bool = False) -> None:
        self._snapshot_model.reset()
        self._tab_widget.clear()

        port_range = None
        if self._run_model.queue_system == QueueSystem.LOCAL:
            port_range = range(49152, 51819)
        evaluator_server_config = EvaluatorServerConfig(custom_port_range=port_range)

        def run() -> None:
            self._run_model.start_simulations_thread(
                evaluator_server_config=evaluator_server_config,
                restart=restart,
            )

        simulation_thread = ErtThread(
            name="ert_gui_simulation_thread", target=run, daemon=True
        )

        self._worker_thread = QThread(parent=self)
        self.destroyed.connect(lambda: _stop_worker(self))

        self._worker = QueueEmitter(self._event_queue)
        self._worker.done.connect(self._worker_thread.quit)
        self._worker.new_event.connect(self._on_event)
        self._worker.moveToThread(self._worker_thread)

        self.simulation_done.connect(self._worker.stop)

        self._worker_thread.started.connect(self._worker.consume_and_emit)
        self._ticker.start(self._RUN_TIME_POLL_RATE)

        self._worker_thread.start()
        simulation_thread.start()
        self._notifier.set_is_simulation_running(True)

    def killJobs(self) -> QMessageBox.StandardButton:
        msg = "Are you sure you want to terminate the currently running experiment?"
        kill_job = QMessageBox.question(
            self, "Terminate experiment", msg, QMessageBox.Yes | QMessageBox.No
        )

        if kill_job == QMessageBox.Yes:
            # Normally this slot would be invoked by the signal/slot system,
            # but the worker is busy tracking the evaluation.
            self._run_model.cancel()
            self._on_finished()
            self.finished.emit(-1)
        return kill_job

    @Slot(bool, str)
    def _on_simulation_done(self, failed: bool, msg: str) -> None:
        self.processing_animation.hide()
        self.kill_button.setHidden(True)
        self.restart_button.setVisible(self._run_model.has_failed_realizations())
        self.restart_button.setEnabled(self._run_model.support_restart)
        self._notifier.set_is_simulation_running(False)
        if failed:
            self.update_total_progress(1.0, "Failed")
            self.fail_msg_box = ErtMessageBox("ERT experiment failed!", msg, self)
            self.fail_msg_box.exec_()
        else:
            self.update_total_progress(1.0, "Experiment completed.")

    @Slot()
    def _on_ticker(self) -> None:
        runtime = self._run_model.get_runtime()
        self.running_time.setText(format_running_time(runtime))

        maximum_memory_usage = self._snapshot_model.root.max_memory_usage

        if maximum_memory_usage:
            self.memory_usage.setText(
                f"Maximal realization memory usage: {byte_with_unit(maximum_memory_usage)}"
            )

    @Slot(object)
    def _on_event(self, event: object) -> None:
        if isinstance(event, EndEvent):
            self.simulation_done.emit(event.failed, event.msg)
            self._ticker.stop()
            self.done_button.setHidden(False)
        elif isinstance(event, FullSnapshotEvent):
            if event.snapshot is not None:
                self._snapshot_model._add_snapshot(event.snapshot, str(event.iteration))
            self.update_total_progress(event.progress, event.iteration_label)
            self._progress_widget.update_progress(
                event.status_count, event.realization_count
            )
        elif isinstance(event, SnapshotUpdateEvent):
            if event.snapshot is not None:
                self._snapshot_model._update_snapshot(
                    event.snapshot, str(event.iteration)
                )
            self._progress_widget.update_progress(
                event.status_count, event.realization_count
            )
            self.update_total_progress(event.progress, event.iteration_label)
        elif isinstance(event, RunModelUpdateBeginEvent):
            iteration = event.iteration
            widget = UpdateWidget(iteration)
            tab_index = self._tab_widget.addTab(widget, f"Update {iteration}")

            if self._tab_widget.currentIndex() == self._tab_widget.count() - 2:
                self._tab_widget.setCurrentIndex(tab_index)

            widget.begin(event)

        elif isinstance(event, RunModelUpdateEndEvent):
            self._progress_widget.stop_waiting_progress_bar()
            if (widget := self._get_update_widget(event.iteration)) is not None:
                widget.end(event)

        elif (isinstance(event, (RunModelStatusEvent, RunModelTimeEvent))) and (
            widget := self._get_update_widget(event.iteration)
        ) is not None:
            widget.update_status(event)

        elif (isinstance(event, RunModelDataEvent)) and (
            widget := self._get_update_widget(event.iteration)
        ) is not None:
            widget.add_table(event)

        elif isinstance(event, RunModelErrorEvent):
            if (widget := self._get_update_widget(event.iteration)) is not None:
                widget.error(event)

        if (
            isinstance(
                event, (RunModelDataEvent, RunModelUpdateEndEvent, RunModelErrorEvent)
            )
            and self.output_path
        ):
            name = event.name if hasattr(event, "name") else "Report"
            if event.data:
                event.data.to_csv(name, self.output_path / str(event.run_id))

    def _get_update_widget(self, iteration: int) -> UpdateWidget:
        for i in range(0, self._tab_widget.count()):
            widget = self._tab_widget.widget(i)
            if isinstance(widget, UpdateWidget) and widget.iteration == iteration:
                return widget
        raise ValueError("Could not find UpdateWidget")

    def update_total_progress(
        self, progress_value: float, iteration_label: str
    ) -> None:
        progress = int(progress_value * 100)
        if not (0 <= progress <= 100):
            logger = logging.getLogger(__name__)
            logger.warning(f"Total progress bar exceeds [0-100] range: {progress}")
        self._total_progress_bar.setValue(progress)
        self._total_progress_label.setText(
            _TOTAL_PROGRESS_TEMPLATE.format(
                total_progress=progress, iteration_label=iteration_label
            )
        )

    def restart_failed_realizations(self) -> None:
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setText(
            "Note that workflows will only be executed on the restarted "
            "realizations and that this might have unexpected consequences."
        )
        msg.setWindowTitle("Restart failed realizations")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setObjectName("restart_prompt")
        result = msg.exec_()

        if result == QMessageBox.Ok:
            self.restart_button.setVisible(False)
            self.kill_button.setVisible(True)
            self.done_button.setVisible(False)
            self.run_experiment(restart=True)

    def get_runtime(self) -> int:
        return self._run_model.get_runtime()

    def _on_finished(self) -> None:
        for file_dialog in self.findChildren(FileDialog):
            file_dialog.close()

    def keyPressEvent(self, a0: Optional[QKeyEvent]) -> None:
        # QDialog on escape will close without prompting
        # so call self.close() instead
        if a0 is not None and a0.key() == Qt.Key.Key_Escape:
            self.close()
        elif a0 is not None and a0.key() == Qt.Key.Key_F1:
            self.produce_clipboard_debug_info.emit()
        else:
            QDialog.keyPressEvent(self, a0)


# Cannot use a non-static method here as
# it is called when the object is destroyed
# https://stackoverflow.com/questions/16842955
def _stop_worker(run_dialog: RunDialog) -> None:
    if run_dialog._worker_thread.isRunning():
        run_dialog._worker.stop()
        run_dialog._worker_thread.wait(3000)
    if run_dialog._worker_thread.isRunning():
        run_dialog._worker_thread.quit()
        run_dialog._worker_thread.wait(3000)
    if run_dialog._worker_thread.isRunning():
        run_dialog._worker_thread.terminate()
        run_dialog._worker_thread.wait(3000)
