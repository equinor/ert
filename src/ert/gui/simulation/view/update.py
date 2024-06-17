from __future__ import annotations

import math
import time
from datetime import timedelta
from typing import Optional

import humanize
from qtpy.QtCore import Qt, Slot
from qtpy.QtGui import QColor, QKeyEvent, QKeySequence
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ert.analysis.event import DataSection
from ert.ensemble_evaluator import state
from ert.run_models import (
    RunModelEvent,
    RunModelStatusEvent,
    RunModelTimeEvent,
    RunModelUpdateBeginEvent,
    RunModelUpdateEndEvent,
)
from ert.run_models.event import RunModelDataEvent, RunModelErrorEvent


class UpdateLogTable(QTableWidget):
    def __init__(self, data: DataSection, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setColumnCount(len(data.header))
        self.setAlternatingRowColors(True)
        self.setRowCount(len(data.data))
        self.setHorizontalHeaderLabels(data.header)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        horizontal_header = self.horizontalHeader()
        assert horizontal_header is not None
        horizontal_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setSortingEnabled(True)
        for i, row in enumerate(data.data):
            for j, val in enumerate(row):
                self.setItem(i, j, QTableWidgetItem(str(val)))

    def keyPressEvent(self, e: Optional[QKeyEvent]) -> None:
        if e is not None and e.matches(QKeySequence.Copy):
            stream = str()
            for i in self.selectedIndexes():
                item = self.itemFromIndex(i)
                assert item is not None
                stream += item.text()
                stream += "\n" if i.column() == self.columnCount() - 1 else "\t"
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(stream)
            else:
                QMessageBox.critical(
                    None,
                    "Error",
                    "Cannot copy text to clipboard because your system does not have a clipboard",
                    QMessageBox.Ok,
                )
        else:
            super().keyPressEvent(e)


class UpdateWidget(QWidget):
    def __init__(self, iteration: int, parent: Optional[QWidget] = None) -> None:
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

    def _insert_status_message(self, message: str) -> None:
        item = QListWidgetItem()
        item.setText(message)
        item.setFlags(item.flags() & ~Qt.ItemFlags(Qt.ItemFlag.ItemIsEnabled))
        self._msg_list.addItem(item)

    def _insert_table_tab(
        self,
        name: str,
        data: DataSection,
    ) -> None:
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        table = UpdateLogTable(data)
        table.setObjectName("CSV_" + name)
        layout.addWidget(table)

        if data.extra:
            grid_layout = QGridLayout()
            nr_each_column = math.ceil(len(data.extra) / 2)
            for i, (k, v) in enumerate(data.extra.items()):
                column = (i // nr_each_column) * 2
                grid_layout.addWidget(QLabel(str(k) + ":"), i % nr_each_column, column)
                grid_layout.addWidget(QLabel(str(v)), i % nr_each_column, column + 1)
            layout.addSpacing(10)
            layout.addLayout(grid_layout)

        self._tab_widget.setCurrentIndex(self._tab_widget.addTab(widget, name))

    @Slot(RunModelUpdateBeginEvent)
    def begin(self, event: RunModelUpdateBeginEvent) -> None:
        self._start_time = time.perf_counter()

    @Slot(RunModelUpdateEndEvent)
    def end(self, event: RunModelUpdateEndEvent) -> None:
        self._progress_msg.setText(
            f"Update completed ({humanize.precisedelta(timedelta(seconds=time.perf_counter() - self._start_time))})"
        )
        self._progress_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background: {QColor(*state.COLOR_FINISHED).name()}; }}"
        )
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(1)
        self._progress_bar.setValue(1)

        self._insert_table_tab("Report", event.data)

    @Slot(RunModelDataEvent)
    def add_table(self, event: RunModelDataEvent) -> None:
        self._insert_table_tab(event.name, event.data)

    @Slot(RunModelEvent)
    def update_status(self, event: RunModelEvent) -> None:
        if isinstance(event, RunModelStatusEvent):
            self._insert_status_message(event.msg)
        elif isinstance(event, RunModelTimeEvent):
            self._progress_msg.setText(
                f"Estimated remaining time for current step {event.remaining_time:.2f}s"
            )

    @Slot(RunModelErrorEvent)
    def error(self, event: RunModelErrorEvent) -> None:
        if event.error_msg:
            self._insert_status_message(f"Error: {event.error_msg}")

        self._progress_msg.setText(
            f"Update failed ({humanize.precisedelta(timedelta(seconds=time.perf_counter() - self._start_time))})"
        )
        self._progress_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background: {QColor(*state.COLOR_FAILED).name()}; }}"
        )
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(1)
        self._progress_bar.setValue(1)

        if (d := event.data) is not None:
            self._insert_table_tab("Report", d)
