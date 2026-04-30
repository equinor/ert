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

from _ert.events import (
    WorkflowBatchFinishedEvent,
    WorkflowCancelledEvent,
    WorkflowFinishedEvent,
    WorkflowStartedEvent,
    WorkflowStatus,
)
from _ert.hook_runtime import HookRuntime
from ert.ensemble_evaluator import state

_STATUS_TO_BACKGROUND = {
    WorkflowStatus.PENDING: QColor(*state.COLOR_PENDING),
    WorkflowStatus.RUNNING: QColor(*state.COLOR_RUNNING),
    WorkflowStatus.FINISHED: QColor(*state.COLOR_FINISHED),
    WorkflowStatus.FAILED: QColor(*state.COLOR_FAILED),
    WorkflowStatus.CANCELLED: QColor(*state.COLOR_CANCELLED),
}
WORKFLOW_JOB_NAME_COLUMN = 0
STATUS_COLUMN = 1
STDOUT_COLUMN = 2
STDERR_COLUMN = 3

HEADER_TO_COLUMN = {
    "WORKFLOW": WORKFLOW_JOB_NAME_COLUMN,
    "STATUS": STATUS_COLUMN,
    "STDOUT": STDOUT_COLUMN,
    "STDERR": STDERR_COLUMN,
}


class WorkflowWidget(QWidget):
    def __init__(
        self,
        hook: HookRuntime,
        workflow_names: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.hook = hook
        self._rows_by_name: dict[str, list[int]] = {}

        self._status_label = QLabel(f"{self.hook.workflow_tab_title()} running")

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(list(HEADER_TO_COLUMN.keys()))
        self._table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerItem)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setMouseTracking(True)

        horizontal_header = self._table.horizontalHeader()
        assert horizontal_header is not None
        horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)

        vertical_header = self._table.verticalHeader()
        assert vertical_header is not None
        vertical_header.setMinimumWidth(20)

        self.setMinimumHeight(140)

        layout = QVBoxLayout(self)
        layout.addWidget(self._status_label)
        layout.addWidget(self._table)

        for row, workflow_name in enumerate(workflow_names):
            self._add_row(row, workflow_name, WorkflowStatus.PENDING)
        self._resize_columns()

    def handle_event(
        self,
        event: WorkflowStartedEvent
        | WorkflowFinishedEvent
        | WorkflowCancelledEvent
        | WorkflowBatchFinishedEvent,
    ) -> None:
        match event:
            case WorkflowStartedEvent(workflow_name=workflow_name):
                row = self._match_first_row_with_status(
                    workflow_name, WorkflowStatus.PENDING
                )
                assert row is not None
                self._set_status(row, WorkflowStatus.RUNNING)
            case WorkflowFinishedEvent(
                workflow_name=workflow_name,
                status=status,
                stdout=stdout,
                stderr=stderr,
            ):
                assert status in {WorkflowStatus.FINISHED, WorkflowStatus.FAILED}
                row = self._match_first_row_with_status(
                    workflow_name,
                    status=WorkflowStatus.RUNNING,
                )
                assert row is not None
                self._set_output(row, stdout, stderr)
                self._set_status(row, status)
            case WorkflowCancelledEvent(workflow_name=workflow_name):
                row = self._match_first_row_with_status(
                    workflow_name,
                    status=WorkflowStatus.RUNNING,
                )
                assert row is not None
                self._set_status(row, WorkflowStatus.CANCELLED)
                self._cancel_pending_rows()
                self._status_label.setText(
                    f"{self.hook.workflow_tab_title()} cancelled"
                )
            case WorkflowBatchFinishedEvent(status=status):
                assert status in {
                    WorkflowStatus.FINISHED,
                    WorkflowStatus.FAILED,
                    WorkflowStatus.CANCELLED,
                }
                self._status_label.setText(
                    f"{self.hook.workflow_tab_title()} {status.value.lower()}"
                )

    def workflow_status(self, workflow_name: str) -> list[WorkflowStatus]:
        workflow_status = []
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, HEADER_TO_COLUMN["WORKFLOW"])
            assert name_item is not None
            if name_item.text() == workflow_name:
                status_item = self._table.item(row, HEADER_TO_COLUMN["STATUS"])
                assert status_item is not None
                workflow_status.append(WorkflowStatus(status_item.text()))
        return workflow_status

    def _set_status(
        self,
        row: int,
        status: WorkflowStatus,
    ) -> None:
        status_item = self._table.item(row, HEADER_TO_COLUMN["STATUS"])
        assert status_item is not None
        status_item.setText(status.value)
        self._apply_row_style(row, status)

    def _match_first_row_with_status(
        self,
        workflow_name: str,
        status: WorkflowStatus,
    ) -> int | None:

        for row in range(self._table.rowCount()):
            status_item = self._table.item(row, HEADER_TO_COLUMN["STATUS"])
            name_item = self._table.item(row, HEADER_TO_COLUMN["WORKFLOW"])
            assert name_item is not None
            assert status_item is not None
            if status_item.text() == status.value and name_item.text() == workflow_name:
                return row

        raise KeyError(
            f"No row found for workflow: {workflow_name} with status: {status}"
        )

    def _add_row(self, row: int, workflow_name: str, status: WorkflowStatus) -> None:
        self._table.insertRow(row)
        self._table.setItem(
            row, HEADER_TO_COLUMN["WORKFLOW"], QTableWidgetItem(workflow_name)
        )
        self._table.setItem(
            row, HEADER_TO_COLUMN["STATUS"], QTableWidgetItem(status.value)
        )
        self._table.setItem(row, HEADER_TO_COLUMN["STDOUT"], QTableWidgetItem())
        self._table.setItem(row, HEADER_TO_COLUMN["STDERR"], QTableWidgetItem())

    def _set_output(self, row: int, stdout: str | None, stderr: str | None) -> None:
        stderr_item = self._table.item(row, HEADER_TO_COLUMN["STDERR"])
        stdout_item = self._table.item(row, HEADER_TO_COLUMN["STDOUT"])
        assert stdout_item is not None
        assert stderr_item is not None
        self._set_output_item(stdout_item, stdout)
        self._set_output_item(stderr_item, stderr)

    def _cancel_pending_rows(self) -> None:
        for row in range(self._table.rowCount()):
            status_item = self._table.item(row, HEADER_TO_COLUMN["Status"])
            assert status_item is not None
            if status_item.text() == WorkflowStatus.PENDING.value:
                self._set_status(row, WorkflowStatus.CANCELLED)

    def _apply_row_style(self, row: int, status: WorkflowStatus) -> None:
        background = _STATUS_TO_BACKGROUND.get(status, QColor(*state.COLOR_UNKNOWN))
        for column in range(self._table.columnCount()):
            item = self._table.item(row, column)
            assert item is not None
            item.setBackground(background or QColor())

    def _set_output_item(self, item: QTableWidgetItem, text: str | None) -> None:
        # None text means it is still pending, empty
        # string means it has been run but has no output
        item.setText("-" if text == "" else text)  # noqa: PLC1901
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setToolTip(text)

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
