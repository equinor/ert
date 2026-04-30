import pytest

from _ert.events import (
    WorkflowBatchFinishedEvent,
    WorkflowBatchStartedEvent,
    WorkflowFinishedEvent,
    WorkflowStartedEvent,
)
from _ert.hook_runtime import HookRuntime
from ert.gui.experiments.view.workflow import WorkflowWidget


def test_workflow_widget_can_be_initialized_without_workflows(qtbot):
    widget = WorkflowWidget(
        HookRuntime.PRE_EXPERIMENT,
        ["prepare_case", "seed_data"],
    )
    qtbot.addWidget(widget)

    assert widget.summary_text() == "Pre-experiment workflows running"
    assert widget.workflow_status("prepare_case") == "Pending"
    assert widget.workflow_status("seed_data") == "Pending"
    assert widget.workflow_output("prepare_case") == ("-", "-")


def test_workflow_widget_rejects_batch_started_event(qtbot):
    widget = WorkflowWidget(HookRuntime.POST_EXPERIMENT, ["archive_results"])
    qtbot.addWidget(widget)

    with pytest.raises(ValueError, match="should create WorkflowWidget"):
        widget.handle_event(
            WorkflowBatchStartedEvent(
                hook=HookRuntime.POST_EXPERIMENT,
                workflow_names=["archive_results"],
            )
        )


def test_workflow_widget_updates_known_workflow_event(qtbot):
    widget = WorkflowWidget(HookRuntime.POST_EXPERIMENT, ["archive_results"])
    qtbot.addWidget(widget)

    widget.handle_event(
        WorkflowStartedEvent(
            hook=HookRuntime.POST_EXPERIMENT,
            workflow_name="archive_results",
        )
    )

    widget.handle_event(
        WorkflowFinishedEvent(
            hook=HookRuntime.POST_EXPERIMENT,
            workflow_name="archive_results",
            status="failure",
            stdout=None,
            stderr="archive failed",
        )
    )

    assert widget.workflow_status("archive_results") == "Failed"
    assert widget.workflow_output("archive_results") == ("-", "archive failed")


def test_workflow_widget_reuses_first_duplicate_row_and_skips_incomplete_rows(qtbot):
    widget = WorkflowWidget(HookRuntime.PRE_UPDATE, ["PRINT", "PRINT"])
    qtbot.addWidget(widget)

    widget.handle_event(
        WorkflowStartedEvent(hook=HookRuntime.PRE_UPDATE, workflow_name="PRINT")
    )
    widget.handle_event(
        WorkflowFinishedEvent(
            hook=HookRuntime.PRE_UPDATE,
            workflow_name="PRINT",
            status="success",
            stdout="first stdout",
            stderr="",
        )
    )
    widget.handle_event(
        WorkflowStartedEvent(hook=HookRuntime.PRE_UPDATE, workflow_name="PRINT")
    )
    widget.handle_event(
        WorkflowFinishedEvent(
            hook=HookRuntime.PRE_UPDATE,
            workflow_name="PRINT",
            status="failure",
            stdout="second stdout",
            stderr="second stderr",
        )
    )

    widget.handle_event(
        WorkflowFinishedEvent(
            hook=HookRuntime.PRE_UPDATE,
            workflow_name="PRINT",
            status="success",
            stdout="updated stdout",
            stderr="",
        )
    )

    assert widget.workflow_output("PRINT", occurrence=0) == ("updated stdout", "-")
    assert widget.workflow_output("PRINT", occurrence=1) == (
        "second stdout",
        "second stderr",
    )

    widget._table.takeItem(0, 0)
    widget.handle_event(
        WorkflowBatchFinishedEvent(
            hook=HookRuntime.PRE_UPDATE,
            workflow_names=["PRINT", "PRINT"],
            status="success",
        )
    )

    assert widget.summary_text() == "Pre-update workflows completed"


@pytest.mark.parametrize(
    ("batch_status", "expected_second", "expected_pending", "expected_summary"),
    [
        (
            "success",
            "Finished",
            "Finished",
            "Pre-update workflows completed",
        ),
        ("failure", "Not run", "Not run", "Pre-update workflows failed"),
    ],
)
def test_workflow_widget_finishes_duplicate_and_pending_rows(
    qtbot,
    batch_status,
    expected_second,
    expected_pending,
    expected_summary,
):
    widget = WorkflowWidget(HookRuntime.PRE_UPDATE, ["PRINT", "PRINT", "ARCHIVE"])
    qtbot.addWidget(widget)

    widget.handle_event(
        WorkflowStartedEvent(hook=HookRuntime.PRE_UPDATE, workflow_name="PRINT")
    )
    widget.handle_event(
        WorkflowFinishedEvent(
            hook=HookRuntime.PRE_UPDATE,
            workflow_name="PRINT",
            status="success",
            stdout="first stdout",
            stderr="",
        )
    )
    widget.handle_event(
        WorkflowStartedEvent(hook=HookRuntime.PRE_UPDATE, workflow_name="PRINT")
    )

    widget.handle_event(
        WorkflowBatchFinishedEvent(
            hook=HookRuntime.PRE_UPDATE,
            workflow_names=["PRINT", "PRINT", "ARCHIVE"],
            status=batch_status,
        )
    )

    assert widget.workflow_status("PRINT", occurrence=0) == "Finished"
    assert widget.workflow_status("PRINT", occurrence=1) == expected_second
    assert widget.workflow_status("ARCHIVE") == expected_pending
    assert widget.workflow_output("PRINT", occurrence=0) == ("first stdout", "-")
    assert widget.summary_text() == expected_summary
