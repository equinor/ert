import pytest
from PyQt6.QtGui import QColor
from pytestqt.qtbot import QtBot

from _ert.events import (
    WorkflowBatchFinishedEvent,
    WorkflowBatchStartedEvent,
    WorkflowCancelledEvent,
    WorkflowFinishedEvent,
    WorkflowStartedEvent,
    WorkflowStatus,
)
from _ert.hook_runtime import HookRuntime
from ert.ensemble_evaluator import state
from ert.gui.experiments.view.workflow import (
    STATUS_COLUMN,
    STDERR_COLUMN,
    STDOUT_COLUMN,
    WORKFLOW_JOB_NAME_COLUMN,
    WorkflowWidget,
    workflow_tab_title,
)

ALL_WORKFLOW_NAMES = ["generate files", "delete files", "archive files"]

WORKFLOW_BATCH_START_EVENT = WorkflowBatchStartedEvent(
    hook=HookRuntime.PRE_EXPERIMENT,
    workflow_names=ALL_WORKFLOW_NAMES,
)

GENERATE_FILES_WORKFLOW_START_EVENT = WorkflowStartedEvent(
    hook=HookRuntime.PRE_EXPERIMENT,
    workflow_name="generate files",
)

DELETE_FILES_WORKFLOW_START_EVENT = WorkflowStartedEvent(
    hook=HookRuntime.PRE_EXPERIMENT,
    workflow_name="delete files",
)

GENERATE_FILES_WORKFLOW_FINISH_EVENT = WorkflowFinishedEvent(
    hook=HookRuntime.PRE_EXPERIMENT,
    workflow_name="generate files",
    status=WorkflowStatus.FINISHED,
    stdout="",
    stderr="",
)

GENERATE_FILES_WORKFLOW_FAIL_EVENT = WorkflowFinishedEvent(
    hook=HookRuntime.PRE_EXPERIMENT,
    workflow_name="generate files",
    status=WorkflowStatus.FAILED,
    stdout="",
    stderr="PermissionError[FooBar]",
)

DELETE_FILES_WORKFLOW_FINISH_EVENT = WorkflowFinishedEvent(
    hook=HookRuntime.PRE_EXPERIMENT,
    workflow_name="delete files",
    status=WorkflowStatus.FINISHED,
    stdout="Deleted successfully",
    stderr="",
)

GENERATE_FILES_WORKFLOW_CANCEL_EVENT = WorkflowCancelledEvent(
    hook=HookRuntime.PRE_EXPERIMENT,
    workflow_name="generate files",
    stdout=None,
    stderr=None,
)

WORKFLOW_BATCH_FINISH_EVENT = WorkflowBatchFinishedEvent(
    hook=HookRuntime.PRE_EXPERIMENT,
    workflow_names=ALL_WORKFLOW_NAMES,
    status=WorkflowStatus.FINISHED,
)


def _get_row_data(workflow_widget: WorkflowWidget, row: int, column: int) -> str:
    item = workflow_widget._table.item(row, column)
    assert item is not None
    return item.text()


@pytest.mark.parametrize(
    ("hook", "expected_label"),
    [
        (HookRuntime.PRE_EXPERIMENT, "Pre-experiment workflows running"),
        (HookRuntime.POST_EXPERIMENT, "Post-experiment workflows running"),
        (HookRuntime.PRE_SIMULATION, "Pre-simulation workflows running"),
        (HookRuntime.POST_SIMULATION, "Post-simulation workflows running"),
        (HookRuntime.PRE_FIRST_UPDATE, "Pre-first-update workflows running"),
        (HookRuntime.PRE_UPDATE, "Pre-update workflows running"),
        (HookRuntime.POST_UPDATE, "Post-update workflows running"),
    ],
)
def test_that_workflow_widget_updates_label_correctly_on_new_events(
    qtbot: QtBot,
    hook: HookRuntime,
    expected_label: str,
) -> None:
    widget = WorkflowWidget(hook)
    qtbot.addWidget(widget)

    assert widget._status_label.text() == expected_label
    widget.handle_event(
        WorkflowBatchStartedEvent(
            hook=hook,
            workflow_names=ALL_WORKFLOW_NAMES,
        )
    )
    widget.handle_event(
        WorkflowBatchFinishedEvent(
            hook=hook,
            workflow_names=ALL_WORKFLOW_NAMES,
            status=WorkflowStatus.FINISHED,
        )
    )
    assert widget._status_label.text() == f"{workflow_tab_title(hook)} finished"

    widget.handle_event(
        WorkflowBatchFinishedEvent(
            hook=hook,
            workflow_names=ALL_WORKFLOW_NAMES,
            status=WorkflowStatus.FAILED,
        )
    )
    assert widget._status_label.text() == f"{workflow_tab_title(hook)} failed"

    widget.handle_event(
        WorkflowBatchFinishedEvent(
            hook=hook,
            workflow_names=ALL_WORKFLOW_NAMES,
            status=WorkflowStatus.CANCELLED,
        )
    )
    assert widget._status_label.text() == f"{workflow_tab_title(hook)} cancelled"


def test_that_workflow_widget_initializes_with_correct_headers(qtbot: QtBot) -> None:
    widget = WorkflowWidget(HookRuntime.PRE_EXPERIMENT)
    qtbot.addWidget(widget)
    assert widget._table.columnCount() == 4
    assert widget._status_label.text() == "Pre-experiment workflows running"
    assert widget._table.horizontalHeaderItem(0).text() == "WORKFLOW"
    assert widget._table.horizontalHeaderItem(1).text() == "STATUS"
    assert widget._table.horizontalHeaderItem(2).text() == "STDOUT"
    assert widget._table.horizontalHeaderItem(3).text() == "STDERR"


def test_that_workflow_widget_adds_all_workflows_as_pending_on_started_batch(
    qtbot: QtBot,
) -> None:
    widget = WorkflowWidget(HookRuntime.PRE_EXPERIMENT)
    qtbot.addWidget(widget)
    assert widget._table.rowCount() == 0
    start_batch_event = WORKFLOW_BATCH_START_EVENT

    widget.handle_event(start_batch_event)
    assert widget._table.rowCount() == len(start_batch_event.workflow_names)
    for row, workflow_name in enumerate(start_batch_event.workflow_names):
        assert _get_row_data(widget, row, WORKFLOW_JOB_NAME_COLUMN) == workflow_name
        assert _get_row_data(widget, row, STATUS_COLUMN) == WorkflowStatus.PENDING.value
        assert _get_row_data(widget, row, STDOUT_COLUMN) == ""  # noqa: PLC1901
        assert _get_row_data(widget, row, STDERR_COLUMN) == ""  # noqa: PLC1901


def test_that_workflows_give_correct_tooltip_on_hover(qtbot: QtBot) -> None:
    widget = WorkflowWidget(HookRuntime.PRE_EXPERIMENT)
    qtbot.addWidget(widget)
    widget.handle_event(WORKFLOW_BATCH_START_EVENT)

    widget.handle_event(GENERATE_FILES_WORKFLOW_START_EVENT)
    widget.handle_event(GENERATE_FILES_WORKFLOW_FAIL_EVENT)
    widget.handle_event(DELETE_FILES_WORKFLOW_START_EVENT)
    widget.handle_event(DELETE_FILES_WORKFLOW_FINISH_EVENT)

    stdout_item = widget._table.item(0, STDOUT_COLUMN)
    stderr_item = widget._table.item(0, STDERR_COLUMN)
    assert stdout_item is not None
    assert stderr_item is not None
    assert not stdout_item.toolTip()
    assert stderr_item.toolTip() == "PermissionError[FooBar]"

    stdout_item = widget._table.item(1, STDOUT_COLUMN)
    stderr_item = widget._table.item(1, STDERR_COLUMN)
    assert stdout_item is not None
    assert stderr_item is not None
    assert stdout_item.toolTip() == "Deleted successfully"
    assert not stderr_item.toolTip()


def test_that_workflow_widget_keeps_output_cells_empty_while_running(
    qtbot: QtBot,
) -> None:
    widget = WorkflowWidget(HookRuntime.PRE_EXPERIMENT)
    qtbot.addWidget(widget)

    widget.handle_event(WORKFLOW_BATCH_START_EVENT)
    widget.handle_event(GENERATE_FILES_WORKFLOW_START_EVENT)

    assert _get_row_data(widget, 0, STATUS_COLUMN) == WorkflowStatus.RUNNING.value
    assert _get_row_data(widget, 0, STDOUT_COLUMN) == ""  # noqa: PLC1901
    assert _get_row_data(widget, 0, STDERR_COLUMN) == ""  # noqa: PLC1901


def test_that_workflow_widget_gives_dash_for_empty_outputs_after_finished_workflow(
    qtbot: QtBot,
) -> None:
    widget = WorkflowWidget(
        HookRuntime.PRE_EXPERIMENT,
    )
    qtbot.addWidget(widget)

    widget.handle_event(WORKFLOW_BATCH_START_EVENT)
    widget.handle_event(GENERATE_FILES_WORKFLOW_START_EVENT)
    widget.handle_event(GENERATE_FILES_WORKFLOW_FINISH_EVENT)

    assert _get_row_data(widget, 0, STDOUT_COLUMN) == "-"
    assert _get_row_data(widget, 0, STDERR_COLUMN) == "-"


@pytest.mark.parametrize(
    ("events", "expected_status", "expected_color"),
    [
        pytest.param(
            [],
            WorkflowStatus.PENDING,
            QColor(*state.COLOR_PENDING),
            id="pending",
        ),
        pytest.param(
            [GENERATE_FILES_WORKFLOW_START_EVENT],
            WorkflowStatus.RUNNING,
            QColor(*state.COLOR_RUNNING),
            id="running",
        ),
        pytest.param(
            [
                GENERATE_FILES_WORKFLOW_START_EVENT,
                GENERATE_FILES_WORKFLOW_FAIL_EVENT,
            ],
            WorkflowStatus.FAILED,
            QColor(*state.COLOR_FAILED),
            id="failed",
        ),
        pytest.param(
            [GENERATE_FILES_WORKFLOW_START_EVENT, GENERATE_FILES_WORKFLOW_CANCEL_EVENT],
            WorkflowStatus.FAILED,
            QColor(*state.COLOR_FAILED),
            id="cancelled",
        ),
        pytest.param(
            [
                GENERATE_FILES_WORKFLOW_START_EVENT,
                GENERATE_FILES_WORKFLOW_FINISH_EVENT,
            ],
            WorkflowStatus.FINISHED,
            QColor(*state.COLOR_FINISHED),
            id="finished",
        ),
    ],
)
def test_that_workflow_widget_paints_rows_based_on_status(
    qtbot: QtBot,
    events: list[WorkflowStartedEvent | WorkflowFinishedEvent | WorkflowCancelledEvent],
    expected_status: WorkflowStatus,
    expected_color: QColor,
) -> None:
    widget = WorkflowWidget(HookRuntime.PRE_EXPERIMENT)
    qtbot.addWidget(widget)

    start_batch_event = WORKFLOW_BATCH_START_EVENT
    first_workflow_name = start_batch_event.workflow_names[0]

    widget.handle_event(start_batch_event)

    for event in events:
        widget.handle_event(event)
    widget.handle_event(WORKFLOW_BATCH_FINISH_EVENT)

    assert _get_row_data(widget, 0, WORKFLOW_JOB_NAME_COLUMN) == first_workflow_name
    assert _get_row_data(widget, 0, STATUS_COLUMN) == expected_status.value

    # All columns in the row should have the same color based on the workflow status
    for column in range(widget._table.columnCount()):
        item = widget._table.item(0, column)
        assert item is not None
        assert item.background().color() == expected_color


def test_that_workflow_widget_updates_on_workflow_event(qtbot: QtBot) -> None:
    widget = WorkflowWidget(HookRuntime.PRE_EXPERIMENT)
    qtbot.addWidget(widget)

    widget.handle_event(WORKFLOW_BATCH_START_EVENT)
    widget.handle_event(GENERATE_FILES_WORKFLOW_START_EVENT)

    assert _get_row_data(widget, 0, STATUS_COLUMN) == WorkflowStatus.RUNNING.value
    assert _get_row_data(widget, 0, STDOUT_COLUMN) == ""  # noqa: PLC1901
    assert _get_row_data(widget, 0, STDERR_COLUMN) == ""  # noqa: PLC1901

    widget.handle_event(GENERATE_FILES_WORKFLOW_FAIL_EVENT)

    assert _get_row_data(widget, 0, STATUS_COLUMN) == WorkflowStatus.FAILED.value
    assert _get_row_data(widget, 0, STDOUT_COLUMN) == "-"
    assert _get_row_data(widget, 0, STDERR_COLUMN) == "PermissionError[FooBar]"


def test_that_updating_one_workflow_does_not_affect_other_rows(qtbot: QtBot) -> None:
    widget = WorkflowWidget(
        HookRuntime.PRE_EXPERIMENT,
    )
    qtbot.addWidget(widget)

    widget.handle_event(WORKFLOW_BATCH_START_EVENT)
    widget.handle_event(GENERATE_FILES_WORKFLOW_START_EVENT)
    widget.handle_event(GENERATE_FILES_WORKFLOW_FAIL_EVENT)
    widget.handle_event(DELETE_FILES_WORKFLOW_START_EVENT)
    widget.handle_event(DELETE_FILES_WORKFLOW_FINISH_EVENT)

    assert _get_row_data(widget, 0, STATUS_COLUMN) == WorkflowStatus.FAILED.value
    assert _get_row_data(widget, 0, STDOUT_COLUMN) == "-"
    assert _get_row_data(widget, 0, STDERR_COLUMN) == "PermissionError[FooBar]"

    assert _get_row_data(widget, 1, STATUS_COLUMN) == WorkflowStatus.FINISHED.value
    assert _get_row_data(widget, 1, STDOUT_COLUMN) == "Deleted successfully"
    assert _get_row_data(widget, 1, STDERR_COLUMN) == "-"

    assert _get_row_data(widget, 2, STATUS_COLUMN) == WorkflowStatus.PENDING.value
    assert _get_row_data(widget, 2, STDOUT_COLUMN) == ""  # noqa: PLC1901
    assert _get_row_data(widget, 2, STDERR_COLUMN) == ""  # noqa: PLC1901


def test_that_workflow_cancelled_event_sets_remaining_jobs_as_cancelled(
    qtbot: QtBot,
) -> None:
    widget = WorkflowWidget(
        HookRuntime.PRE_EXPERIMENT,
    )
    qtbot.addWidget(widget)
    widget.handle_event(WORKFLOW_BATCH_START_EVENT)
    widget.handle_event(GENERATE_FILES_WORKFLOW_START_EVENT)
    widget.handle_event(GENERATE_FILES_WORKFLOW_CANCEL_EVENT)

    assert _get_row_data(widget, 0, STATUS_COLUMN) == WorkflowStatus.FAILED.value
    assert _get_row_data(widget, 0, STDERR_COLUMN) == "Cancelled by user"

    assert _get_row_data(widget, 1, STATUS_COLUMN) == WorkflowStatus.CANCELLED.value
    assert _get_row_data(widget, 2, STATUS_COLUMN) == WorkflowStatus.CANCELLED.value

    assert widget._status_label.text() == "Pre-experiment workflows cancelled"


@pytest.mark.parametrize(
    ("status", "expected_label"),
    [
        pytest.param(WorkflowStatus.FAILED, "failed", id="failed"),
        pytest.param(WorkflowStatus.CANCELLED, "cancelled", id="cancelled"),
    ],
)
def test_that_early_non_finished_workflow_batch_marks_pending_jobs_as_cancelled(
    qtbot: QtBot,
    status: WorkflowStatus,
    expected_label: str,
) -> None:
    widget = WorkflowWidget(HookRuntime.PRE_EXPERIMENT)
    qtbot.addWidget(widget)

    widget.handle_event(WORKFLOW_BATCH_START_EVENT)
    widget.handle_event(GENERATE_FILES_WORKFLOW_START_EVENT)
    widget.handle_event(GENERATE_FILES_WORKFLOW_FAIL_EVENT)
    widget.handle_event(
        WorkflowBatchFinishedEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_names=ALL_WORKFLOW_NAMES,
            status=status,
        )
    )

    assert _get_row_data(widget, 0, STATUS_COLUMN) == WorkflowStatus.FAILED.value
    assert _get_row_data(widget, 1, STATUS_COLUMN) == WorkflowStatus.CANCELLED.value
    assert _get_row_data(widget, 2, STATUS_COLUMN) == WorkflowStatus.CANCELLED.value
    assert widget._status_label.text() == f"Pre-experiment workflows {expected_label}"
