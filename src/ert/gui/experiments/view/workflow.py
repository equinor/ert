from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from _ert.hook_runtime import HookRuntime
from ert.ensemble_evaluator import state

_STATUS_TO_BACKGROUND = {
    "Pending": QColor(*state.COLOR_PENDING),
    "Running": QColor(*state.COLOR_RUNNING),
    "Finished": QColor(*state.COLOR_FINISHED),
    "Failed": QColor(*state.COLOR_FAILED),
    "Not run": QColor(*state.COLOR_CANCELLED),
}
_ROW_FOREGROUND = QColor(Qt.GlobalColor.black)


def workflow_tab_title(hook: HookRuntime, iteration: int | None = None) -> str:
    title = {
        HookRuntime.PRE_EXPERIMENT: "Pre-experiment workflows",
        HookRuntime.POST_EXPERIMENT: "Post-experiment workflows",
        HookRuntime.PRE_SIMULATION: "Pre-simulation workflows",
        HookRuntime.POST_SIMULATION: "Post-simulation workflows",
        HookRuntime.PRE_FIRST_UPDATE: "Pre-first-update workflows",
        HookRuntime.PRE_UPDATE: "Pre-update workflows",
        HookRuntime.POST_UPDATE: "Post-update workflows",
    }[hook]
    if iteration is None:
        return title

    return f"{title} for iteration {iteration}"


class WorkflowWidget(QWidget):
    def __init__(
        self,
        hook: HookRuntime,
        iteration: int | None = None,
        workflow_names: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._hook = hook
        self._iteration = iteration
        self._rows_by_name: dict[str, list[int]] = {}

        self._status_label = QLabel(f"{workflow_tab_title(hook, iteration)} queued")

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(
            ["Workflow job", "Status", "Stdout", "Stderr"]
        )
        self._table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerItem)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setMouseTracking(True)

        horizontal_header = self._table.horizontalHeader()
        assert horizontal_header is not None
        self._resize_columns()

        vertical_header = self._table.verticalHeader()
        assert vertical_header is not None
        vertical_header.setMinimumWidth(20)

        self.setMinimumHeight(140)

        layout = QVBoxLayout(self)
        layout.addWidget(self._status_label)
        layout.addWidget(self._table)

        if workflow_names:
            self.set_workflows(workflow_names)

    @property
    def hook(self) -> HookRuntime:
        return self._hook

    @property
    def iteration(self) -> int | None:
        return self._iteration

    def set_workflows(self, workflow_names: list[str]) -> None:
        self._rows_by_name.clear()
        self._table.setRowCount(0)
        for workflow_name in workflow_names:
            self._set_row(workflow_name, "Pending")
        self._resize_columns()
        self._status_label.setText(
            f"{workflow_tab_title(self._hook, self._iteration)} running"
        )

    def start_workflow(self, workflow_name: str) -> None:
        row = self._match_row(workflow_name, preferred_statuses=("Pending",))
        self._set_status(row, "Running")

    def finish_workflow(
        self,
        workflow_name: str,
        status: str,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        row = self._match_row(workflow_name, preferred_statuses=("Running", "Pending"))
        self._set_output(row, stdout or "", stderr or "")
        if status == "success":
            self._set_status(row, "Finished")
        else:
            self._set_status(row, "Failed")

    def finish(self, status: str) -> None:
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, 0)
            status_item = self._table.item(row, 1)
            if name_item is None or status_item is None:
                continue

            if status_item.text() in {"Pending", "Running"}:
                if status == "success":
                    self._set_status(row, "Finished")
                else:
                    self._set_status(row, "Not run")

        outcome = "completed" if status == "success" else "failed"
        self._status_label.setText(
            f"{workflow_tab_title(self._hook, self._iteration)} {outcome}"
        )

    def workflow_status(self, workflow_name: str, occurrence: int = 0) -> str:
        row = self._row_for_name(workflow_name, occurrence)
        item = self._table.item(row, 1)
        assert item is not None
        return item.text()

    def summary_text(self) -> str:
        return self._status_label.text()

    def workflow_output(
        self, workflow_name: str, occurrence: int = 0
    ) -> tuple[str, str]:
        row = self._row_for_name(workflow_name, occurrence)
        stdout_item = self._table.item(row, 2)
        stderr_item = self._table.item(row, 3)
        assert stdout_item is not None
        assert stderr_item is not None
        return stdout_item.text(), stderr_item.text()

    def _set_status(
        self,
        row: int,
        status_text: str,
    ) -> None:
        status_item = self._table.item(row, 1)
        assert status_item is not None
        status_item.setText(status_text)
        self._apply_row_style(row, status_text)

    def _row_for_name(self, workflow_name: str, occurrence: int) -> int:
        rows = self._rows_by_name[workflow_name]
        return rows[occurrence]

    def _match_row(
        self,
        workflow_name: str,
        preferred_statuses: tuple[str, ...],
    ) -> int:
        rows = self._rows_by_name.get(workflow_name)
        if rows is None:
            return self._set_row(workflow_name, "Pending")

        for row in rows:
            status_item = self._table.item(row, 1)
            assert status_item is not None
            if status_item.text() in preferred_statuses:
                return row

        return rows[0]

    def _set_row(self, workflow_name: str, status_text: str) -> int:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._rows_by_name.setdefault(workflow_name, []).append(row)
        self._table.setItem(row, 0, QTableWidgetItem(workflow_name))
        self._table.setItem(row, 1, QTableWidgetItem(status_text))
        stdout_item = QTableWidgetItem()
        stderr_item = QTableWidgetItem()
        self._set_output_item(stdout_item, "")
        self._set_output_item(stderr_item, "")
        self._table.setItem(row, 2, stdout_item)
        self._table.setItem(row, 3, stderr_item)
        self._apply_row_style(row, status_text)
        return row

    def _set_output(self, row: int, stdout: str, stderr: str) -> None:
        stdout_item = self._table.item(row, 2)
        stderr_item = self._table.item(row, 3)
        assert stdout_item is not None
        assert stderr_item is not None
        self._set_output_item(stdout_item, stdout)
        self._set_output_item(stderr_item, stderr)
        self._resize_columns()

    def _apply_row_style(self, row: int, status_text: str) -> None:
        background = _STATUS_TO_BACKGROUND.get(status_text)
        for column in range(self._table.columnCount()):
            item = self._table.item(row, column)
            assert item is not None
            item.setForeground(_ROW_FOREGROUND)
            item.setBackground(background or QColor())

    def _set_output_item(self, item: QTableWidgetItem, text: str) -> None:
        if text:
            item.setText(text)
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            item.setToolTip(text)
            return

        item.setText("-")
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setToolTip("")

    def _resize_columns(self) -> None:
        horizontal_header = self._table.horizontalHeader()
        assert horizontal_header is not None

        horizontal_header.resizeSections(QHeaderView.ResizeMode.ResizeToContents)
        for section in range(horizontal_header.count()):
            if horizontal_header.sectionSize(section) < 135:
                horizontal_header.resizeSection(section, 135)

            horizontal_header.setSectionResizeMode(
                section,
                QHeaderView.ResizeMode.Stretch
                if section == horizontal_header.count() - 1
                else QHeaderView.ResizeMode.Interactive,
            )
