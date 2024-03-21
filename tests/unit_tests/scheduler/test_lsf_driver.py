import asyncio
import os
import stat
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from textwrap import dedent
from typing import Collection, List, get_args

import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.utils import poll

from ert.scheduler import LsfDriver
from ert.scheduler.lsf_driver import (
    BSUB_FLAKY_SSH,
    LSF_FAILED_JOB,
    FinishedEvent,
    FinishedJobFailure,
    FinishedJobSuccess,
    JobState,
    QueuedJob,
    RunningJob,
    StartedEvent,
    _Stat,
    parse_bjobs,
)

valid_jobstates: Collection[str] = list(get_args(JobState))


def nonempty_string_without_whitespace():
    return st.text(
        st.characters(whitelist_categories=("Lu", "Ll", "Nd", "P")), min_size=1
    )


@pytest.fixture
def capturing_bsub(monkeypatch, tmp_path):
    os.chdir(tmp_path)
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bsub_path = bin_path / "bsub"
    bsub_path.write_text(
        "#!/bin/sh\n"
        "echo $@ > captured_bsub_args\n"
        "echo 'Job <1>' is submitted to normal queue",
        encoding="utf-8",
    )
    bsub_path.chmod(bsub_path.stat().st_mode | stat.S_IEXEC)


@given(st.lists(st.sampled_from(JobState.__args__)))
async def test_events_produced_from_jobstate_updates(jobstate_sequence: List[str]):

    started = any(
        state in jobstate_sequence
        for state in RunningJob.model_json_schema()["properties"]["job_state"]["enum"]
    )
    finished_success = any(
        state in jobstate_sequence
        for state in FinishedJobSuccess.model_json_schema()["properties"]["job_state"][
            "enum"
        ]
    )
    finished_failure = any(
        state in jobstate_sequence
        for state in FinishedJobFailure.model_json_schema()["properties"]["job_state"][
            "enum"
        ]
    )

    driver = LsfDriver()

    async def mocked_submit(self, iens, *_args, **_kwargs):
        """A mocked submit is speedier than going through a command on disk"""
        self._jobs["1"] = (iens, QueuedJob(job_state="PEND"))
        self._iens2jobid[iens] = "1"

    driver.submit = mocked_submit.__get__(driver)
    await driver.submit(0, "_")

    # Replicate the behaviour of multiple calls to poll()
    for statestr in jobstate_sequence:
        jobstate = _Stat(**{"jobs": {"1": {"job_state": statestr}}}).jobs["1"]
        await driver._process_job_update("1", jobstate)

    events = []
    while not driver.event_queue.empty():
        events.append(await driver.event_queue.get())

    if not started and not finished_success and not finished_failure:
        assert len(events) == 0

        iens, state = driver._jobs["1"]
        assert iens == 0
        assert isinstance(state, QueuedJob)
    elif started and not finished_success and not finished_failure:
        assert len(events) == 1
        assert events[0] == StartedEvent(iens=0)

        iens, state = driver._jobs["1"]
        assert iens == 0
        assert isinstance(state, RunningJob)
    elif started and finished_success and finished_failure:
        assert len(events) <= 2  # The StartedEvent is not required
        assert events[-1] == FinishedEvent(iens=0, returncode=events[-1].returncode)
        assert "1" not in driver._jobs
    elif started is True and finished_success and not finished_failure:
        assert len(events) <= 2  # The StartedEvent is not required
        assert events[-1] == FinishedEvent(iens=0, returncode=0)
        assert "1" not in driver._jobs
    elif started is True and not finished_success and finished_failure:
        assert len(events) <= 2  # The StartedEvent is not required
        assert events[-1] == FinishedEvent(iens=0, returncode=LSF_FAILED_JOB)
        assert "1" not in driver._jobs


@pytest.mark.usefixtures("capturing_bsub")
async def test_submit_with_named_queue():
    driver = LsfDriver(queue_name="foo")
    await driver.submit(0, "sleep")
    assert "-q foo" in Path("captured_bsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_bsub")
async def test_submit_with_default_queue():
    driver = LsfDriver()
    await driver.submit(0, "sleep")
    assert "-q" not in Path("captured_bsub_args").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "bsub_script, expectation",
    [
        pytest.param(
            "echo 'Job <444> is submitted to default queue'",
            does_not_raise(),
            id="all-good",
        ),
        pytest.param(
            "echo 'Job <444> is subtted to default queue'",
            pytest.raises(RuntimeError, match="subtte"),
            id="typo_from_bsub",
        ),
        pytest.param(
            "echo 'Job <0> is submitted to default queue'",
            does_not_raise(),
            id="zero_job_id",
        ),
        pytest.param(
            "echo 'Job <-1> is submitted to default queue'",
            pytest.raises(RuntimeError, match="Could not understand"),
            id="negative_job_id",
        ),
        pytest.param(
            "echo 'Job <fuzz> is submitted to default queue'",
            pytest.raises(RuntimeError, match="Could not understand"),
            id="non_number_job_id",
        ),
        pytest.param(
            "echo 'Job <1.4> is submitted to default queue'",
            pytest.raises(RuntimeError, match="Could not understand"),
            id="floating_job_id",
        ),
        pytest.param("exit 1", pytest.raises(RuntimeError), id="plain_exit_1"),
        pytest.param(
            "echo no_go >&2; exit 1",
            pytest.raises(RuntimeError, match="no_go"),
            id="exit_with_stderr",
        ),
        pytest.param(
            "exit 0", pytest.raises(RuntimeError), id="exit_0_but_empty_stdout"
        ),
    ],
)
async def test_faulty_bsub(monkeypatch, tmp_path, bsub_script, expectation):
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bsub_path = bin_path / "bsub"
    bsub_path.write_text(f"#!/bin/sh\n{bsub_script}")
    bsub_path.chmod(bsub_path.stat().st_mode | stat.S_IEXEC)
    driver = LsfDriver()
    with expectation:
        await driver.submit(0, "sleep")


@pytest.mark.parametrize(
    "mocked_iens2jobid, iens_to_kill, bkill_returncode, bkill_stdout, bkill_stderr, expected_logged_error",
    [
        pytest.param(
            {"1": "11"},
            "1",
            0,
            "Job <11> is being terminated",
            "",
            None,
            id="happy_path",
        ),
        pytest.param(
            {"1": "11"},
            "2",
            1,
            "",
            "",
            "LSF kill failed due to missing",
            id="internal_ert_error",
        ),
        pytest.param(
            {"1": "11"},
            "1",
            255,
            "",
            "Job <22>: No matching job found",
            "No matching job found",
            id="inconsistency_ert_vs_lsf",
        ),
        pytest.param(
            {"1": "11"},
            "1",
            0,
            "wrong_stdout...",
            "",
            "wrong_stdout...",
            id="artifical_bkill_stdout_giving_logged_error",
        ),
        pytest.param(
            {"1": "11"},
            "1",
            0,
            "",
            "wrong_on_stderr",
            "wrong_on_stderr",
            id="artifical_bkill_stderr_giving_logged_error",
        ),
        pytest.param(
            {"1": "11"},
            "1",
            1,
            "",
            "wrong_on_stderr",
            "wrong_on_stderr",
            id="artifical_bkill_stderr_and_returncode_giving_logged_error",
        ),
    ],
)
async def test_kill(
    monkeypatch,
    tmp_path,
    mocked_iens2jobid,
    iens_to_kill,
    bkill_returncode,
    bkill_stdout,
    bkill_stderr,
    expected_logged_error,
    caplog,
):
    os.chdir(tmp_path)
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bkill_path = bin_path / "bkill"
    bkill_path.write_text(
        f"#!/bin/sh\necho '{bkill_stdout}'\n"
        f"echo '{bkill_stderr}' >&2\n"
        f"echo $@ > 'bkill_args'\n"
        f"exit {bkill_returncode}",
        encoding="utf-8",
    )
    bkill_path.chmod(bkill_path.stat().st_mode | stat.S_IEXEC)

    driver = LsfDriver()

    driver._iens2jobid = mocked_iens2jobid
    await driver.kill(iens_to_kill)
    if expected_logged_error:
        assert expected_logged_error in caplog.text
    else:
        assert (
            mocked_iens2jobid[iens_to_kill]
            == Path("bkill_args").read_text(encoding="utf-8").strip()
        )


@given(st.text())
def test_parse_bjobs_gives_empty_result_on_random_input(some_text):
    assert parse_bjobs(some_text) == {"jobs": {}}


@pytest.mark.parametrize(
    "bjobs_output, expected",
    [
        pytest.param(
            "JOBID   USER   STAT\n1 foobart RUN",
            {"1": {"job_state": "RUN"}},
            id="basic",
        ),
        pytest.param(
            "1 foobart RUN", {"1": {"job_state": "RUN"}}, id="header_missing_ok"
        ),
        pytest.param(
            "1 _ RUN asdf asdf asdf",
            {"1": {"job_state": "RUN"}},
            id="line_remainder_ignored",
        ),
        pytest.param("1 _ DONE", {"1": {"job_state": "DONE"}}, id="done"),
        pytest.param(
            "1 _ DONE\n2 _ RUN",
            {"1": {"job_state": "DONE"}, "2": {"job_state": "RUN"}},
            id="two_jobs",
        ),
    ],
)
def test_parse_bjobs_happy_path(bjobs_output, expected):
    assert parse_bjobs(bjobs_output) == {"jobs": expected}


@given(
    st.integers(min_value=1),
    nonempty_string_without_whitespace(),
    st.from_type(JobState),
)
def test_parse_bjobs(job_id, username, job_state):
    assert parse_bjobs(f"{job_id} {username} {job_state}") == {
        "jobs": {str(job_id): {"job_state": job_state}}
    }


def test_parse_bjobs_handles_output_with_exec_host_split_over_two_lines():
    bjobs_output = (
        "JOBID   USER    STAT  QUEUE      FROM_HOST   EXEC_HOST   JOB_NAME   SUBMIT_TIME\n"
        "479460  xxxx    RUN   allcpus    foo-host-n1 3*foo-host1 FOO_00-0   Feb 14 13:07\n"
        "                                             4*foo-hostn105-05-10"
    )
    assert parse_bjobs(bjobs_output) == {"jobs": {"479460": {"job_state": "RUN"}}}


def test_parse_bjobs_handles_output_with_no_exec_host():
    bjobs_output = (
        "JOBID   USER    STAT  QUEUE      FROM_HOST   EXEC_HOST   JOB_NAME   SUBMIT_TIME\n"
        "479460  xxxx    RUN   allcpus    foo-host-n1             FOO_00-0   Feb 14 13:07\n"
    )
    assert parse_bjobs(bjobs_output) == {"jobs": {"479460": {"job_state": "RUN"}}}


@given(nonempty_string_without_whitespace().filter(lambda x: x not in valid_jobstates))
def test_parse_bjobs_invalid_state_is_ignored(random_state):
    assert parse_bjobs(f"1 _ {random_state}") == {"jobs": {}}


def test_parse_bjobs_invalid_state_is_logged(caplog):
    # (cannot combine caplog with hypothesis)
    parse_bjobs("1 _ FOO")
    assert "Unknown state FOO" in caplog.text


BJOBS_HEADER = (
    "JOBID   USER    STAT  QUEUE      FROM_HOST   EXEC_HOST   JOB_NAME   SUBMIT_TIME"
)


@pytest.mark.parametrize(
    "bjobs_script, expectation",
    [
        pytest.param(
            f"echo '{BJOBS_HEADER}\n1 someuser DONE foo'; exit 0",
            does_not_raise(),
            id="all-good",
        ),
        pytest.param(
            "echo 'No unfinished job found'; exit 0",
            pytest.raises(asyncio.TimeoutError),
            id="empty_cluster",
        ),
        pytest.param(
            "echo 'Job <1> is not found' >&2; exit 0",
            # Actual command is seen to return zero in such a scenario
            pytest.raises(asyncio.TimeoutError),
            id="empty_cluster_specific_id",
        ),
        pytest.param(
            "echo '1 someuser DONE foo'",
            does_not_raise(),
            id="missing_header_is_accepted",  # (debatable)
        ),
        pytest.param(
            f"echo '{BJOBS_HEADER}\n1 someuser DONE foo'; "
            "echo 'Job <2> is not found' >&2 ; exit 255",
            # If we have some success and some failures, actual command returns 255
            does_not_raise(),
            id="error_for_irrelevant_job_id",
        ),
        pytest.param(
            f"echo '{BJOBS_HEADER}\n2 someuser DONE foo'",
            pytest.raises(asyncio.TimeoutError),
            id="wrong-job-id",
        ),
        pytest.param(
            "exit 1",
            pytest.raises(asyncio.TimeoutError),
            id="exit-1",
        ),
        pytest.param(
            f"echo '{BJOBS_HEADER}\n1 someuser DONE foo'; exit 1",
            # (this is not observed in reality)
            does_not_raise(),
            id="correct_output_but_exitcode_1",
        ),
        pytest.param(
            f"echo '{BJOBS_HEADER}\n1 someuser'; exit 0",
            pytest.raises(asyncio.TimeoutError),
            id="unparsable_output",
        ),
    ],
)
async def test_faulty_bjobs(monkeypatch, tmp_path, bjobs_script, expectation):
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bsub_path = bin_path / "bsub"
    bsub_path.write_text("#!/bin/sh\necho 'Job <1> is submitted to default queue'")
    bsub_path.chmod(bsub_path.stat().st_mode | stat.S_IEXEC)
    bjobs_path = bin_path / "bjobs"
    bjobs_path.write_text(f"#!/bin/sh\n{bjobs_script}")
    bjobs_path.chmod(bjobs_path.stat().st_mode | stat.S_IEXEC)
    driver = LsfDriver()
    with expectation:
        await driver.submit(0, "sleep")
        await asyncio.wait_for(poll(driver, {0}), timeout=0.2)


@pytest.mark.parametrize(
    ("exit_code, error_msg"),
    [
        (BSUB_FLAKY_SSH, ""),
        (199, "Not recognized"),
    ],
)
async def test_that_bsub_will_retry_and_fail(
    monkeypatch, tmp_path, exit_code, error_msg
):
    os.chdir(tmp_path)
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bsub_path = bin_path / "bsub"
    bsub_path.write_text(f"#!/bin/sh\necho {error_msg} >&2\nexit {exit_code}")
    bsub_path.chmod(bsub_path.stat().st_mode | stat.S_IEXEC)
    driver = LsfDriver()
    driver._bsub_retries = 2
    driver._sleep_time_between_cmd_retries = 0.2
    match_str = (
        f"failed after 2 retries with error {error_msg}"
        if exit_code != 199
        else "failed with exit code 199 and error message: Not recognized"
    )
    with pytest.raises(RuntimeError, match=match_str):
        await driver.submit(0, "sleep 10")


@pytest.mark.parametrize(
    ("exit_code, error_msg"),
    [
        (BSUB_FLAKY_SSH, ""),
    ],
)
async def test_that_bsub_will_retry_and_succeed(
    monkeypatch, tmp_path, exit_code, error_msg
):
    os.chdir(tmp_path)
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bsub_path = bin_path / "bsub"
    bsub_path.write_text(
        "#!/bin/sh"
        + dedent(
            f"""
            TRY_FILE="{bin_path}/script_try"
            if [ -f "$TRY_FILE" ]; then
                echo "Job <1> is submitted to normal queue"
                exit 0
            else
                echo "TRIED" > $TRY_FILE
                echo "{error_msg}" >&2
                exit {exit_code}
            fi
            """
        )
    )
    bsub_path.chmod(bsub_path.stat().st_mode | stat.S_IEXEC)
    driver = LsfDriver()
    driver._bsub_retries = 2
    driver._sleep_time_between_cmd_retries = 0.2
    await driver.submit(0, "sleep 10")
