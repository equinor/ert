from PyQt6.QtGui import QColor
from pytestqt.qtbot import QtBot

from _ert.events import (
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
    return workflow_widget._table.item(row, column).text()


def test_that_workflow_widget_updates_label_correctly_on_new_events(
    qtbot: QtBot,
) -> None:
    pass


# parametrize hook runtime to test the title label is correct
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
    pass


def test_that_workflow_widget_gives_empty_string_for_none_outputs(qtbot: QtBot) -> None:
    """Workflows with no stdout/stderr should show empty when it is
    still running, but there is no output yet"""


def test_that_workflow_widget_gives_dash_for_empty_outputs(qtbot: QtBot) -> None:
    """Workflows that have completed but have no output should show a
    dash to indicate they have been run"""


def test_that_workflow_widget_paints_rows_based_on_status(qtbot: QtBot) -> None:
    pending_job_name = "pending_job"
    running_job_name = "running_job"
    failed_job_name = "failed_job"
    cancelled_job_name = "cancelled_job"
    finished_job_name = "finished_job"
    widget = WorkflowWidget(
        HookRuntime.PRE_EXPERIMENT,
        [
            pending_job_name,
            running_job_name,
            failed_job_name,
            cancelled_job_name,
            finished_job_name,
        ],
    )
    qtbot.addWidget(widget)

    widget.handle_event(
        WorkflowStartedEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name=running_job_name,
        )
    )
    widget.handle_event(
        WorkflowFinishedEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name=failed_job_name,
            status=WorkflowStatus.FAILED,
            stdout=None,
            stderr=None,
        )
    )
    widget.handle_event(
        WorkflowFinishedEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name=finished_job_name,
            status=WorkflowStatus.FINISHED,
            stdout=None,
            stderr=None,
        )
    )
    widget.handle_event(
        WorkflowCancelledEvent(
            hook=HookRuntime.PRE_EXPERIMENT,
            workflow_name=cancelled_job_name,
            stdout=None,
            stderr=None,
        )
    )

    def assert_row_name_status_and_color(
        row: int, expected_name: str, expected_status: str, expected_color: QColor
    ):
        assert _get_row_data(widget, row, WORKFLOW_JOB_NAME_COLUMN) == expected_name
        assert _get_row_data(widget, row, STATUS_COLUMN) == expected_status
        for column in range(widget._table.columnCount()):
            item = widget._table.item(row, column)
            assert item is not None
            assert item.background().color() == expected_color

    assert_row_name_status_and_color(
        0, pending_job_name, WorkflowStatus.PENDING.value, QColor(*state.COLOR_PENDING)
    )
    assert_row_name_status_and_color(
        1, running_job_name, WorkflowStatus.RUNNING.value, QColor(*state.COLOR_RUNNING)
    )
    assert_row_name_status_and_color(
        2, failed_job_name, WorkflowStatus.FAILED.value, QColor(*state.COLOR_FAILED)
    )
    assert_row_name_status_and_color(
        3,
        cancelled_job_name,
        WorkflowStatus.CANCELLED.value,
        QColor(*state.COLOR_CANCELLED),
    )
    assert_row_name_status_and_color(
        4,
        finished_job_name,
        WorkflowStatus.FINISHED.value,
        QColor(*state.COLOR_FINISHED),
    )


def test_that_workflow_widget_updates_on_workflow_event(qtbot: QtBot):
    widget = WorkflowWidget(HookRuntime.PRE_EXPERIMENT, ["generate files"])
    qtbot.addWidget(widget)

    widget.handle_event(
        WorkflowStartedEvent(
            hook=HookRuntime.POST_EXPERIMENT,
            workflow_name="generate files",
        )
    )
    assert _get_row_data(widget, 0, STATUS_COLUMN) == WorkflowStatus.PENDING.value
    assert _get_row_data(widget, 0, STDOUT_COLUMN) == ""  # noqa: PLC1901
    assert _get_row_data(widget, 0, STDERR_COLUMN) == ""  # noqa: PLC1901
    widget.handle_event(
        WorkflowFinishedEvent(
            hook=HookRuntime.POST_EXPERIMENT,
            workflow_name="generate files",
            status=WorkflowStatus.FAILED,
            stdout="",
            stderr="failed to generate files",
        )
    )

    assert _get_row_data(widget, 0, STATUS_COLUMN) == WorkflowStatus.FAILED.value
    assert _get_row_data(widget, 0, STDOUT_COLUMN) == "-"
    assert _get_row_data(widget, 0, STDERR_COLUMN) == "failed to generate files"


def test_that_workflow_batch_finished_event_sets_remaining_jobs_as_cancelled():
    """In case something happens that causes the remaining workflows to never
    be ran, they should be marked as non-pending/cancelled"""
