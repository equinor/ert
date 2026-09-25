from __future__ import annotations

from typing import cast

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFontDatabase
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ert.ensemble_evaluator.state import COLOR_CANCELLED, COLOR_FAILED, COLOR_FINISHED
from ert.run_models.event import WorkflowEvent
from ert.workflow_runner import WorkflowJobStatus

NO_ITERATION_LABEL = "Pre/post experiment"
NO_OUTPUT_PLACEHOLDER = "(no output)"
_EXPERIMENT_WIDE_HOOKS = frozenset({"PRE_EXPERIMENT", "POST_EXPERIMENT"})

_COLUMNS = ("Hook", "Workflow", "Job", "Status", "Time")


class WorkflowLogWidget(QWidget):
    # Table of workflow job invocations with their captured output.

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._events: dict[int | None, list[WorkflowEvent]] = {}
        self._iteration_chosen_by_user = False

        self._iteration_selector = QComboBox(self)
        self._iteration_selector.currentIndexChanged.connect(self._on_iteration_changed)
        self._iteration_selector.activated.connect(self._on_iteration_chosen_by_user)

        selector_row = QHBoxLayout()
        selector_row.setSpacing(6)
        selector_row.addWidget(QLabel("Iteration:"))
        selector_row.addWidget(self._iteration_selector)
        selector_row.addStretch()

        self._table = QTableWidget(0, len(_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        vertical_header = self._table.verticalHeader()
        assert vertical_header is not None
        vertical_header.setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self._table.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            _COLUMNS.index("Job"), QHeaderView.ResizeMode.Stretch
        )
        self._table.itemSelectionChanged.connect(self._on_row_selected)

        self._stdout_view = self._make_output_view()
        self._stderr_view = self._make_output_view()

        detail = QSplitter(Qt.Orientation.Horizontal, self)
        detail.addWidget(self._make_labelled_output("Stdout", self._stdout_view))
        detail.addWidget(self._make_labelled_output("Stderr", self._stderr_view))

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.addWidget(self._table)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(selector_row)
        layout.addWidget(splitter)

        self._clear_detail()

    def _make_output_view(self) -> QPlainTextEdit:
        view = QPlainTextEdit(self)
        view.setReadOnly(True)
        view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        return view

    def _make_labelled_output(self, title: str, view: QPlainTextEdit) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(QLabel(title, container))
        layout.addWidget(view)
        return container

    def add_event(self, event: WorkflowEvent) -> None:
        group = self._group_key(event)
        is_new_group = group not in self._events
        self._events.setdefault(group, []).append(event)

        if is_new_group:
            self._rebuild_iteration_selector()
        elif group == self._selected_iteration():
            self._append_row(event)

    def _group_key(self, event: WorkflowEvent) -> int | None:
        if event.hook in _EXPERIMENT_WIDE_HOOKS:
            return None
        return event.iteration

    def _selected_iteration(self) -> int | None:
        index = self._iteration_selector.currentIndex()
        if index < 0:
            return None
        return cast(int | None, self._iteration_selector.itemData(index))

    def _rebuild_iteration_selector(self) -> None:
        previously_selected = self._selected_iteration()
        had_selection = self._iteration_selector.count() > 0

        iterations: list[int | None] = []
        if None in self._events:
            iterations.append(None)
        iterations.extend(sorted(i for i in self._events if i is not None))

        self._iteration_selector.blockSignals(True)
        self._iteration_selector.clear()
        for iteration in iterations:
            label = (
                NO_ITERATION_LABEL if iteration is None else f"Iteration {iteration}"
            )
            self._iteration_selector.addItem(label, iteration)
        if (
            self._iteration_chosen_by_user
            and had_selection
            and previously_selected in iterations
        ):
            self._iteration_selector.setCurrentIndex(
                iterations.index(previously_selected)
            )
        else:
            # Follow the newest iteration until the user picks one themselves.
            self._iteration_selector.setCurrentIndex(len(iterations) - 1)
        self._iteration_selector.blockSignals(False)
        self._on_iteration_changed()

    def _on_iteration_chosen_by_user(self, _index: int) -> None:
        self._iteration_chosen_by_user = True

    def _on_iteration_changed(self) -> None:
        self._table.clearContents()
        self._table.setRowCount(0)
        for event in self._events.get(self._selected_iteration(), []):
            self._append_row(event)
        self._clear_detail()

    def _append_row(self, event: WorkflowEvent) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        job = event.job_name
        if event.arguments:
            job += f"({', '.join(event.arguments)})"
        status, color = self._status_and_color(event)

        values = (
            event.hook,
            event.workflow_name,
            job,
            status,
            event.timestamp.strftime("%H:%M:%S"),
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 3:
                item.setBackground(color)
                item.setForeground(QColor(0, 0, 0))
            self._table.setItem(row, column, item)

    def _status_and_color(self, event: WorkflowEvent) -> tuple[str, QColor]:
        match event.status:
            case WorkflowJobStatus.CANCELLED:
                return "Cancelled", QColor(*COLOR_CANCELLED)
            case WorkflowJobStatus.FAILED:
                return "Failed", QColor(*COLOR_FAILED)
            case WorkflowJobStatus.SUCCESS:
                return "Succeeded", QColor(*COLOR_FINISHED)

    def _on_row_selected(self) -> None:
        selection_model = self._table.selectionModel()
        assert selection_model is not None
        rows = selection_model.selectedRows()
        if not rows:
            self._clear_detail()
            return
        events = self._events.get(self._selected_iteration(), [])
        row = rows[0].row()
        if row >= len(events):
            self._clear_detail()
            return
        event = events[row]
        self._stdout_view.setPlainText(event.stdout or NO_OUTPUT_PLACEHOLDER)
        self._stderr_view.setPlainText(event.stderr or NO_OUTPUT_PLACEHOLDER)

    def _clear_detail(self) -> None:
        self._stdout_view.setPlainText(NO_OUTPUT_PLACEHOLDER)
        self._stderr_view.setPlainText(NO_OUTPUT_PLACEHOLDER)

    def clear(self) -> None:
        """Discard all displayed workflow events, e.g. before a rerun."""
        self._events = {}
        self._iteration_chosen_by_user = False
        self._iteration_selector.blockSignals(True)
        self._iteration_selector.clear()
        self._iteration_selector.blockSignals(False)
        self._table.clearContents()
        self._table.setRowCount(0)
        self._clear_detail()
