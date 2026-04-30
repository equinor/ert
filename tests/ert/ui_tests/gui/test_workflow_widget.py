import pytest
from PyQt6.QtGui import QColor
from pytestqt.qtbot import QtBot

from _ert.events import (
    WorkflowBatchFinishedEvent,
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
)


def _get_row_data(workflow_widget: WorkflowWidget, row: int, column: int) -> str:
    item = workflow_widget._table.item(row, column)
    assert item is not None
    return item.text()


def _assert_row_color(widget: WorkflowWidget, row: int, color: QColor) -> None:
    for column in range(widget._table.columnCount()):
        item = widget._table.item(row, column)
        assert item is not None
        assert item.background().color() == color


@pytest.mark.parametrize(
    ("hook", "expected_label"),
    [
        (HookRuntime.PRE_EXPERIMENT, "Pre-experiment workflows running"),
        (HookRuntime.PRE_SIMULATION, "Pre-simulation workflows running"),
        (HookRuntime.POST_EXPERIMENT, "Post-experiment workflows running"),
    ],
)
def test_that_workflow_widget_updates_label_correctly_on_new_events(
    qtbot: QtBot,
    hook: HookRuntime,
    expected_label: str,
) -> None:
    widget = WorkflowWidget(hook, ["generate files"])
    qtbot.addWidget(widget)

    assert widget._status_label.text() == expected_label

    widget.handle_event(
        WorkflowBatchFinishedEvent(
            hook=hook,
            workflow_names=["generate files"],
            status=WorkflowStatus.FINISHED,
        )
    )
    assert widget._status_label.text() == f"{hook.workflow_tab_title()} finished"

    widget.handle_event(
        WorkflowBatchFinishedEvent(
            hook=hook,
            workflow_names=["generate files"],
            status=WorkflowStatus.FAILED,
        )
    )
    assert widget._status_label.text() == f"{hook.workflow_tab_title()} failed"

    widget.handle_event(
        WorkflowBatchFinishedEvent(
            hook=hook,
            workflow_names=["generate files"],
            status=WorkflowStatus.CANCELLED,
        )
    )
    assert widget._status_label.text() == f"{hook.workflow_tab_title()} cancelled"


def test_that_workflow_widget_initializes_with_correct_headers(qtbot: QtBot) -> None:
    widget = WorkflowWidget(HookRuntime.PRE_EXPERIMENT, [])
    qtbot.addWidget(widget)
    assert widget._table.columnCount() == 4
    assert widget._status_label.text() == "Pre-experiment workflows running"
    assert widget._table.horizontalHeaderItem(0).text() == "WORKFLOW"
    assert widget._table.horizontalHeaderItem(1).text() == "STATUS"
    assert widget._table.horizontalHeaderItem(2).text() == "STDOUT"
    assert widget._table.horizontalHeaderItem(3).text() == "STDERR"


def test_that_workflow_widget_initializes_with_all_workflows_as_pending(
    qtbot: QtBot,
) -> None:
    widget = WorkflowWidget(
        HookRuntime.PRE_EXPERIMENT, ["generate files", "delete files"]
    )
    qtbot.addWidget(widget)
    assert widget._table.rowCount() == 2
    assert _get_row_data(widget, 0, WORKFLOW_JOB_NAME_COLUMN) == "generate files"
    assert _get_row_data(widget, 1, WORKFLOW_JOB_NAME_COLUMN) == "delete files"
    for row in range(widget._table.rowCount()):
        assert _get_row_data(widget, row, STATUS_COLUMN) == WorkflowStatus.PENDING.value
        assert _get_row_data(widget, row, STDOUT_COLUMN) == ""  # noqa: PLC1901
        assert _get_row_data(widget, row, STDERR_COLUMN) == ""  # noqa: PLC1901


def test_that_workflows_give_tooltip_on_hover(qtbot: QtBot) -> None:
    widget = WorkflowWidget(HookRuntime.PRE_EXPERIMENT, ["generate files"])
    qtbot.addWidget(widget)

    widget.handle_event(
        WorkflowStartedEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name="generate files",
        )
    )
    widget.handle_event(
        WorkflowFinishedEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name="generate files",
            status=WorkflowStatus.FAILED,
            stdout="stdout text",
            stderr="stderr text",
        )
    )

    stdout_item = widget._table.item(0, STDOUT_COLUMN)
    stderr_item = widget._table.item(0, STDERR_COLUMN)
    assert stdout_item is not None
    assert stderr_item is not None
    assert stdout_item.toolTip() == "stdout text"
    assert stderr_item.toolTip() == "stderr text"


def test_that_workflow_widget_keeps_output_cells_empty_while_running(
    qtbot: QtBot,
) -> None:
    widget = WorkflowWidget(HookRuntime.PRE_EXPERIMENT, ["generate files"])
    qtbot.addWidget(widget)

    widget.handle_event(
        WorkflowStartedEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name="generate files",
        )
    )

    assert _get_row_data(widget, 0, STATUS_COLUMN) == WorkflowStatus.RUNNING.value
    assert _get_row_data(widget, 0, STDOUT_COLUMN) == ""  # noqa: PLC1901
    assert _get_row_data(widget, 0, STDERR_COLUMN) == ""  # noqa: PLC1901


def test_that_workflow_widget_gives_dash_for_empty_outputs(qtbot: QtBot) -> None:
    widget = WorkflowWidget(HookRuntime.PRE_EXPERIMENT, ["generate files"])
    qtbot.addWidget(widget)

    widget.handle_event(
        WorkflowStartedEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name="generate files",
        )
    )
    widget.handle_event(
        WorkflowFinishedEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name="generate files",
            status=WorkflowStatus.FINISHED,
            stdout="",
            stderr="",
        )
    )

    assert _get_row_data(widget, 0, STDOUT_COLUMN) == "-"
    assert _get_row_data(widget, 0, STDERR_COLUMN) == "-"


@pytest.mark.parametrize(
    ("workflow_name", "events", "expected_status", "expected_color"),
    [
        pytest.param(
            "pending_job",
            [],
            WorkflowStatus.PENDING,
            QColor(*state.COLOR_PENDING),
            id="pending",
        ),
        pytest.param(
            "running_job",
            [
                WorkflowStartedEvent(
                    hook=HookRuntime.PRE_EXPERIMENT,
                    workflow_name="running_job",
                )
            ],
            WorkflowStatus.RUNNING,
            QColor(*state.COLOR_RUNNING),
            id="running",
        ),
        pytest.param(
            "failed_job",
            [
                WorkflowStartedEvent(
                    hook=HookRuntime.PRE_EXPERIMENT,
                    workflow_name="failed_job",
                ),
                WorkflowFinishedEvent(
                    hook=HookRuntime.PRE_EXPERIMENT,
                    workflow_name="failed_job",
                    status=WorkflowStatus.FAILED,
                    stdout=None,
                    stderr=None,
                ),
            ],
            WorkflowStatus.FAILED,
            QColor(*state.COLOR_FAILED),
            id="failed",
        ),
        pytest.param(
            "cancelled_job",
            [
                WorkflowStartedEvent(
                    hook=HookRuntime.PRE_EXPERIMENT,
                    workflow_name="cancelled_job",
                ),
                WorkflowCancelledEvent(
                    hook=HookRuntime.PRE_EXPERIMENT,
                    workflow_name="cancelled_job",
                    stdout=None,
                    stderr=None,
                ),
            ],
            WorkflowStatus.CANCELLED,
            QColor(*state.COLOR_CANCELLED),
            id="cancelled",
        ),
        pytest.param(
            "finished_job",
            [
                WorkflowStartedEvent(
                    hook=HookRuntime.PRE_EXPERIMENT,
                    workflow_name="finished_job",
                ),
                WorkflowFinishedEvent(
                    hook=HookRuntime.PRE_EXPERIMENT,
                    workflow_name="finished_job",
                    status=WorkflowStatus.FINISHED,
                    stdout=None,
                    stderr=None,
                ),
            ],
            WorkflowStatus.FINISHED,
            QColor(*state.COLOR_FINISHED),
            id="finished",
        ),
    ],
)
def test_that_workflow_widget_paints_rows_based_on_status(
    qtbot: QtBot,
    workflow_name: str,
    events: list[WorkflowStartedEvent | WorkflowFinishedEvent | WorkflowCancelledEvent],
    expected_status: WorkflowStatus,
    expected_color: QColor,
) -> None:
    widget = WorkflowWidget(HookRuntime.PRE_EXPERIMENT, [workflow_name])
    qtbot.addWidget(widget)

    for event in events:
        widget.handle_event(event)

    assert _get_row_data(widget, 0, WORKFLOW_JOB_NAME_COLUMN) == workflow_name
    assert _get_row_data(widget, 0, STATUS_COLUMN) == expected_status.value
    _assert_row_color(widget, 0, expected_color)


def test_that_workflow_widget_updates_on_workflow_event(qtbot: QtBot) -> None:
    widget = WorkflowWidget(HookRuntime.PRE_EXPERIMENT, ["generate files"])
    qtbot.addWidget(widget)

    widget.handle_event(
        WorkflowStartedEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name="generate files",
        )
    )
    assert _get_row_data(widget, 0, STATUS_COLUMN) == WorkflowStatus.RUNNING.value
    assert _get_row_data(widget, 0, STDOUT_COLUMN) == ""  # noqa: PLC1901
    assert _get_row_data(widget, 0, STDERR_COLUMN) == ""  # noqa: PLC1901
    widget.handle_event(
        WorkflowFinishedEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name="generate files",
            status=WorkflowStatus.FAILED,
            stdout="",
            stderr="failed to generate files",
        )
    )

    assert _get_row_data(widget, 0, STATUS_COLUMN) == WorkflowStatus.FAILED.value
    assert _get_row_data(widget, 0, STDOUT_COLUMN) == "-"
    assert _get_row_data(widget, 0, STDERR_COLUMN) == "failed to generate files"


def test_that_updating_one_workflow_does_not_affect_other_rows(qtbot: QtBot) -> None:
    widget = WorkflowWidget(
        HookRuntime.PRE_EXPERIMENT,
        ["generate files", "delete files", "archive files"],
    )
    qtbot.addWidget(widget)

    widget.handle_event(
        WorkflowStartedEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name="delete files",
        )
    )
    widget.handle_event(
        WorkflowFinishedEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name="delete files",
            status=WorkflowStatus.FINISHED,
            stdout="deleted",
            stderr="",
        )
    )

    assert _get_row_data(widget, 0, STATUS_COLUMN) == WorkflowStatus.PENDING.value
    assert _get_row_data(widget, 0, STDOUT_COLUMN) == ""  # noqa: PLC1901
    assert _get_row_data(widget, 0, STDERR_COLUMN) == ""  # noqa: PLC1901

    assert _get_row_data(widget, 1, STATUS_COLUMN) == WorkflowStatus.FINISHED.value
    assert _get_row_data(widget, 1, STDOUT_COLUMN) == "deleted"
    assert _get_row_data(widget, 1, STDERR_COLUMN) == "-"

    assert _get_row_data(widget, 2, STATUS_COLUMN) == WorkflowStatus.PENDING.value
    assert _get_row_data(widget, 2, STDOUT_COLUMN) == ""  # noqa: PLC1901
    assert _get_row_data(widget, 2, STDERR_COLUMN) == ""  # noqa: PLC1901


def test_that_workflow_cancelled_event_sets_remaining_jobs_as_cancelled(
    qtbot: QtBot,
) -> None:
    widget = WorkflowWidget(
        HookRuntime.PRE_EXPERIMENT,
        ["generate files", "delete files", "archive files"],
    )
    qtbot.addWidget(widget)

    widget.handle_event(
        WorkflowStartedEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name="generate files",
        )
    )
    widget.handle_event(
        WorkflowCancelledEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name="generate files",
            stdout=None,
            stderr=None,
        )
    )

    assert _get_row_data(widget, 0, STATUS_COLUMN) == WorkflowStatus.CANCELLED.value
    assert _get_row_data(widget, 1, STATUS_COLUMN) == WorkflowStatus.CANCELLED.value
    assert _get_row_data(widget, 2, STATUS_COLUMN) == WorkflowStatus.CANCELLED.value
    assert widget._status_label.text() == "Pre-experiment workflows cancelled"
