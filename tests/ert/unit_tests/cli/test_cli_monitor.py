from datetime import UTC, datetime, timedelta
from io import StringIO
from queue import SimpleQueue
from uuid import uuid4

from ert.cli.monitor import Monitor
from ert.ensemble_evaluator.event import EndEvent, SnapshotUpdateEvent
from ert.ensemble_evaluator.snapshot import (
    EnsembleSnapshot,
    RealizationSnapshot,
)
from ert.ensemble_evaluator.state import (
    REALIZATION_STATE_FINISHED,
    REALIZATION_STATE_RUNNING,
    REALIZATION_STATE_WAITING,
)
from ert.run_models.event import WorkflowEvent
from ert.workflow_runner import WorkflowJobStatus


def test_color_always():
    out = StringIO()  # not a tty, so coloring is automatically disabled
    monitor = Monitor(out=out, color_always=True)

    assert monitor._colorize("Foo", color=(255, 0, 0)) == "\x1b[38;2;255;0;0mFoo\x1b[0m"


def test_legends():
    monitor = Monitor(out=StringIO())
    snapshot = EnsembleSnapshot()
    snapshot._ensemble_state = ""
    for i in range(100):
        status = REALIZATION_STATE_FINISHED if i < 10 else REALIZATION_STATE_RUNNING
        snapshot.add_realization(
            str(i), RealizationSnapshot(status=status, active=True)
        )
    monitor._snapshots[0] = snapshot
    legends = monitor._get_legends()

    assert (
        legends
        == """    Waiting         0/100
    Pending         0/100
    Running        90/100
    Failed          0/100
    Finished       10/100
    Unknown         0/100
"""
    )


def test_result_success():
    out = StringIO()
    monitor = Monitor(out=out)

    monitor._print_result(False, None)

    assert out.getvalue() == "Experiment completed.\n"


def test_result_failure():
    out = StringIO()
    monitor = Monitor(out=out)

    monitor._print_result(True, "fail")

    assert out.getvalue() == "Experiment failed with the following error: fail\n"


def test_that_monitor_prints_indented_workflow_errors_except_for_stopping_job():
    def workflow_event(
        job_name: str,
        error: str | None,
        status: WorkflowJobStatus,
        *,
        stopped_workflow: bool = False,
    ):
        return WorkflowEvent(
            run_id=uuid4(),
            hook="POST_SIMULATION",
            workflow_name=job_name,
            job_name=job_name,
            job_index=0,
            arguments=[],
            stdout="",
            stderr="printed by job",
            status=status,
            timestamp=datetime.now(tz=UTC),
            error=error,
            stopped_workflow=stopped_workflow,
        )

    events = SimpleQueue()
    events.put(
        workflow_event(
            "FAILING_JOB",
            "ValueError: first line\nsecond line",
            WorkflowJobStatus.FAILED,
        )
    )
    events.put(workflow_event("SUCCEEDING_JOB", None, WorkflowJobStatus.SUCCESS))
    events.put(
        workflow_event(
            "STOPPING_JOB",
            "ValueError: boom",
            WorkflowJobStatus.FAILED,
            stopped_workflow=True,
        )
    )
    events.put(
        EndEvent(
            failed=True,
            msg="Workflow job STOPPING_JOB failed with error: ValueError: boom",
        )
    )
    out = StringIO()

    Monitor(out=out).monitor(events)

    assert out.getvalue() == (
        "Workflow job FAILING_JOB failed: ValueError: first line\n"
        "    second line\n"
        "Experiment failed with the following error: "
        "Workflow job STOPPING_JOB failed with error: ValueError: boom\n"
    )


def test_print_progress():
    out = StringIO()
    monitor = Monitor(out=out)
    snapshot = EnsembleSnapshot()
    snapshot._ensemble_state = ""
    for i in range(100):
        status = REALIZATION_STATE_FINISHED if i < 50 else REALIZATION_STATE_WAITING
        snapshot.add_realization(
            str(i), RealizationSnapshot(status=status, active=True)
        )
    monitor._snapshots[0] = snapshot
    monitor._start_time = datetime.now(tz=UTC)
    general_event = SnapshotUpdateEvent(
        iteration_label="Test Phase",
        total_iterations=2,
        progress=0.5,
        realization_count=100,
        status_count={"Finished": 50, "Waiting": 50},
        iteration=0,
    )

    monitor._print_progress(general_event)

    # For some reason, `tqdm` adds an extra line containing a progress-bar,
    # even though this test only calls it once.
    # I suspect this has something to do with the way `tqdm` does refresh,
    # but do not know how to fix it.
    # Seems not be a an issue when used normally.
    expected = """    --> Test Phase


    |                                                                                      |   0% it
    1/2 |##############################5                              |  50% Running time: 0 seconds

    Waiting        50/100
    Pending         0/100
    Running         0/100
    Failed          0/100
    Finished       50/100
    Unknown         0/100

"""  # ruff: ignore[line-too-long]

    assert out.getvalue().replace("\r", "\n") == expected


def test_that_print_progress_shows_correct_elapsed_time_beyond_24_hours():
    """timedelta.seconds wraps at 86400; total_seconds() gives the true elapsed."""
    out = StringIO()
    monitor = Monitor(out=out)
    snapshot = EnsembleSnapshot()
    snapshot._ensemble_state = ""
    for i in range(10):
        snapshot.add_realization(
            str(i), RealizationSnapshot(status=REALIZATION_STATE_RUNNING, active=True)
        )
    monitor._snapshots[0] = snapshot
    # Simulate a run that has been going for 25 hours
    monitor._start_time = datetime.now(tz=UTC) - timedelta(hours=25)
    event = SnapshotUpdateEvent(
        iteration_label="Long Phase",
        total_iterations=1,
        progress=0.5,
        realization_count=10,
        status_count={"Running": 10},
        iteration=0,
    )

    monitor._print_progress(event)

    output = out.getvalue()
    # The running time line must contain "1 day" (not "1 hour", which is
    # what the buggy timedelta.seconds expression would produce: 25*3600 % 86400
    # = 3600 seconds = 1 hour).
    assert "1 day" in output
    assert "1 hour" in output
