import uuid
from datetime import UTC
from datetime import datetime as dt

import pytest

from ert.run_models.event import RunModelWorkflowLogEvent


def _event(**kwargs) -> RunModelWorkflowLogEvent:
    defaults = {
        "run_id": uuid.uuid4(),
        "hook": "PRE_UPDATE",
        "workflow_name": "my_workflow",
        "job_name": "MY_JOB",
        "job_index": 0,
        "arguments": [],
        "stdout": "",
        "stderr": "",
        "failed": False,
        "timestamp": dt(2020, 1, 1, 12, 0, 0, tzinfo=UTC),
    }
    return RunModelWorkflowLogEvent(**(defaults | kwargs))


def test_that_workflow_log_entry_contains_stdout_and_stderr():
    entry = _event(stdout="on stdout\n", stderr="on stderr\n").as_log_entry()

    assert "--- stdout ---\non stdout\n" in entry
    assert "--- stderr ---\non stderr\n" in entry


def test_that_workflow_log_entry_omits_the_stderr_section_when_stderr_is_empty():
    entry = _event(stdout="on stdout\n", stderr="").as_log_entry()

    assert "--- stdout ---" in entry
    assert "--- stderr ---" not in entry


def test_that_workflow_log_entry_header_states_hook_workflow_job_and_status():
    entry = _event(
        hook="POST_SIMULATION",
        workflow_name="wf",
        job_name="JOB",
        job_index=2,
        failed=True,
        iteration=3,
    ).as_log_entry()

    assert entry.splitlines()[0] == (
        "=== 2020-01-01T12:00:00+00:00 POST_SIMULATION "
        "workflow=wf job=JOB#2 status=failed iteration=3"
    )


def test_that_a_cancelled_job_is_reported_as_cancelled_even_if_not_failed():
    entry = _event(cancelled=True, failed=False).as_log_entry()

    assert "status=cancelled" in entry.splitlines()[0]


def test_that_workflow_log_entries_are_separated_by_a_blank_line():
    first = _event(job_name="FIRST", stdout="first\n").as_log_entry()
    second = _event(job_name="SECOND", stdout="second\n").as_log_entry()

    assert first.endswith("first\n\n")
    assert (first + second).count("--- stdout ---") == 2


@pytest.mark.parametrize("arguments", [[], ["a", "b"]])
def test_that_the_arguments_section_is_only_written_when_the_job_has_arguments(
    arguments,
):
    entry = _event(arguments=arguments).as_log_entry()

    assert ("--- arguments ---\na b" in entry) == bool(arguments)
