import time
from collections import defaultdict
from typing import DefaultDict

import humanize
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ert.analysis import SmootherSnapshot
from ert.analysis._es_update import ObservationStatus
from ert.run_models import (
    RunModelEvent,
    RunModelStatusEvent,
    RunModelTimeEvent,
    RunModelUpdateBeginEvent,
    RunModelUpdateEndEvent,
)


class UpdateWidget(QWidget):
    def __init__(self, iteration: int, parent=None) -> None:
        super().__init__(parent)

        self._iteration = iteration
        self._start_time: float = 0.0

        progress_label = QLabel("Progress:")
        self._progress_msg = QLabel()

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(0)

        widget = QWidget()
        msg_layout = QHBoxLayout()
        widget.setLayout(msg_layout)

        self._msg_list = QListWidget()
        self._msg_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

        msg_layout.addWidget(self._msg_list)

        self._tab_widget = QTabWidget()
        self._tab_widget.addTab(widget, "Status")
        self._tab_widget.setTabBarAutoHide(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(100, 20, 100, 20)

        top_layout = QHBoxLayout()
        top_layout.addWidget(progress_label)
        top_layout.addWidget(self._progress_msg)
        top_layout.addStretch()

        layout.addLayout(top_layout)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._tab_widget)

        self.setLayout(layout)

    @property
    def iteration(self) -> int:
        return self._iteration

    def _insert_report_tab(self, smoother_snapshot: SmootherSnapshot) -> None:
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        update_step = smoother_snapshot.update_step_snapshots

        obs_info: DefaultDict[ObservationStatus, int] = defaultdict(lambda: 0)
        for update in update_step:
            obs_info[update.status] += 1

        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel("Parent ensemble:"), 0, 0)
        grid_layout.addWidget(QLabel(smoother_snapshot.source_case), 0, 1)
        grid_layout.addWidget(QLabel("Target ensemble:"), 1, 0)
        grid_layout.addWidget(QLabel(smoother_snapshot.target_case), 1, 1)
        grid_layout.addWidget(QLabel("Alpha:"), 2, 0)
        grid_layout.addWidget(QLabel(str(smoother_snapshot.alpha)), 2, 1)
        grid_layout.addWidget(QLabel("Global scaling:"), 3, 0)
        grid_layout.addWidget(QLabel(str(smoother_snapshot.global_scaling)), 3, 1)
        grid_layout.addWidget(QLabel("Standard cutoff:"), 4, 0)
        grid_layout.addWidget(QLabel(str(smoother_snapshot.std_cutoff)), 4, 1)
        grid_layout.addWidget(QLabel("Active observations:"), 5, 0)
        grid_layout.addWidget(QLabel(str(obs_info[ObservationStatus.ACTIVE])), 5, 1)
        grid_layout.addWidget(
            QLabel("Deactivated observations - missing respons(es):"), 6, 0
        )
        grid_layout.addWidget(
            QLabel(str(obs_info[ObservationStatus.MISSING_RESPONSE])), 6, 1
        )
        grid_layout.addWidget(
            QLabel("Deactivated observations - ensemble_std > STD_CUTOFF:"), 7, 0
        )
        grid_layout.addWidget(QLabel(str(obs_info[ObservationStatus.STD_CUTOFF])), 7, 1)
        grid_layout.addWidget(QLabel("Deactivated observations - outlier"), 8, 0)
        grid_layout.addWidget(QLabel(str(obs_info[ObservationStatus.OUTLIER])), 8, 1)

        layout.addLayout(grid_layout)
        layout.addSpacing(20)

        table = QTableWidget()
        table.setColumnCount(5)
        table.setAlternatingRowColors(True)
        table.setRowCount(len(update_step))
        table.setRowCount(len(smoother_snapshot.update_step_snapshots))
        table.setHorizontalHeaderLabels(
            ["Response", "Index", "Observed history", "Simulated data", "Status"]
        )
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setStretchLastSection(True)
        table.setColumnWidth(0, 200)

        def cell(content: str):
            item = QTableWidgetItem(content)
            item.setTextAlignment(Qt.AlignCenter)
            return item

        for nr, step in enumerate(smoother_snapshot.update_step_snapshots):
            obs_std = (
                f"{step.obs_std:.3f}"
                if step.obs_scaling == 1
                else f"{step.obs_std * step.obs_scaling:.3f} ({step.obs_std:.3f} * {step.obs_scaling:.3f})"
            )
            table.setItem(
                nr,
                0,
                cell(f"{step.obs_name}"),
            )
            table.setItem(
                nr,
                1,
                cell(f"{step.obs_coord.stringify()}"),
            )
            table.setItem(nr, 2, cell(f"{step.obs_val:.3f} +/- {obs_std}"))
            table.setItem(
                nr,
                3,
                cell(f"{step.response_mean:.3f} +/- {step.response_std:.3f}"),
            )

            table.setItem(nr, 4, cell(f"{step.get_status().capitalize()}"))

            table.setStyleSheet(
                "QTableWidget::item { padding-left: 5px; padding-right: 5px; }"
            )

            table.resizeColumnsToContents()

        layout.addWidget(table)

        self._tab_widget.setCurrentIndex(self._tab_widget.addTab(widget, "Report"))

    @Slot(RunModelUpdateBeginEvent)
    def begin(self, _: RunModelUpdateBeginEvent) -> None:
        self._start_time = time.perf_counter()

    @Slot(RunModelUpdateEndEvent)
    def end(self, event: RunModelUpdateEndEvent) -> None:
        self._progress_msg.setText(
            f"Update completed ({humanize.precisedelta(time.perf_counter() - self._start_time)})"
        )
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(1)
        self._progress_bar.setValue(1)

        if event.smoother_snapshot:
            self._insert_report_tab(event.smoother_snapshot)

    @Slot(RunModelEvent)
    def update_status(self, event: RunModelEvent) -> None:
        if isinstance(event, RunModelStatusEvent):
            item = QListWidgetItem()
            item.setText(event.msg)
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self._msg_list.addItem(item)
        elif isinstance(event, RunModelTimeEvent):
            self._progress_msg.setText(
                f"Estimated remaining time for current step {event.remaining_time:.2f}s"
            )
