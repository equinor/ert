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


def test_that_workflow_log_is_written_to_a_run_id_directory_of_the_output_path(
    tmp_path,
):
    run_id = uuid.uuid4()

    _event(run_id=run_id, stdout="hello\n").write_as_log(tmp_path)

    assert (tmp_path / str(run_id) / "workflows.log").is_file()


def test_that_workflow_log_entry_contains_stdout_and_stderr(tmp_path):
    _event(stdout="on stdout\n", stderr="on stderr\n").write_as_log(tmp_path)

    log = next(tmp_path.glob("*/workflows.log")).read_text(encoding="utf-8")
    assert "--- stdout ---\non stdout\n" in log
    assert "--- stderr ---\non stderr\n" in log


def test_that_workflow_log_entry_omits_the_stderr_section_when_stderr_is_empty(
    tmp_path,
):
    _event(stdout="on stdout\n", stderr="").write_as_log(tmp_path)

    log = next(tmp_path.glob("*/workflows.log")).read_text(encoding="utf-8")
    assert "--- stdout ---" in log
    assert "--- stderr ---" not in log


def test_that_workflow_log_entry_header_states_hook_workflow_job_and_status(tmp_path):
    _event(
        hook="POST_SIMULATION",
        workflow_name="wf",
        job_name="JOB",
        job_index=2,
        failed=True,
        iteration=3,
    ).write_as_log(tmp_path)

    log = next(tmp_path.glob("*/workflows.log")).read_text(encoding="utf-8")
    assert log.splitlines()[0] == (
        "=== 2020-01-01T12:00:00+00:00 POST_SIMULATION "
        "workflow=wf job=JOB#2 status=failed iteration=3"
    )


def test_that_workflow_log_entries_from_the_same_run_are_appended_to_one_file(tmp_path):
    run_id = uuid.uuid4()

    _event(run_id=run_id, job_name="FIRST", stdout="first\n").write_as_log(tmp_path)
    _event(run_id=run_id, job_name="SECOND", stdout="second\n").write_as_log(tmp_path)

    log = (tmp_path / str(run_id) / "workflows.log").read_text(encoding="utf-8")
    assert log.index("job=FIRST") < log.index("job=SECOND")
    assert log.count("--- stdout ---") == 2


def test_that_workflow_log_entries_from_different_runs_are_kept_apart(tmp_path):
    _event(run_id=uuid.uuid4(), stdout="first\n").write_as_log(tmp_path)
    _event(run_id=uuid.uuid4(), stdout="second\n").write_as_log(tmp_path)

    assert len(list(tmp_path.glob("*/workflows.log"))) == 2


def test_that_no_workflow_log_is_written_when_there_is_no_output_path(tmp_path):
    _event(stdout="hello\n").write_as_log(None)

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("arguments", [[], ["a", "b"]])
def test_that_the_arguments_section_is_only_written_when_the_job_has_arguments(
    tmp_path, arguments
):
    _event(arguments=arguments).write_as_log(tmp_path)

    log = next(tmp_path.glob("*/workflows.log")).read_text(encoding="utf-8")
    assert ("--- arguments ---\na b" in log) == bool(arguments)
