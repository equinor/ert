import asyncio
import logging
import os
import stat
import time
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from textwrap import dedent
from typing import Collection, List, Optional, get_args, get_type_hints
from unittest.mock import AsyncMock

import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.utils import poll

from ert.config import QueueConfig
from ert.scheduler import LsfDriver, create_driver
from ert.scheduler.lsf_driver import (
    FLAKY_SSH_RETURNCODE,
    LSF_FAILED_JOB,
    FinishedEvent,
    FinishedJobFailure,
    FinishedJobSuccess,
    JobData,
    JobState,
    QueuedJob,
    RunningJob,
    StartedEvent,
    _parse_jobs_dict,
    build_resource_requirement_string,
    filter_job_ids_on_submission_time,
    parse_bhist,
    parse_bjobs,
)

valid_jobstates: Collection[str] = list(get_args(JobState))


def nonempty_string_without_whitespace():
    return st.text(
        st.characters(whitelist_categories=("Lu", "Ll", "Nd", "P")), min_size=1
    )


@pytest.fixture
def capturing_bsub(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    bin_path = Path("bin")
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


@pytest.mark.parametrize(
    "bjobs_script, bhist_script, exit_code",
    [
        pytest.param("echo 129", "echo Exited with exit code 130", 129),
        pytest.param(
            "exit 9", "echo 'asdfasdf\nExited with exit code 130\nasdfasdf'", 130
        ),
        pytest.param("exit 15", "echo Exited with ex code 130", LSF_FAILED_JOB),
        pytest.param("exit 9", "exit 144", LSF_FAILED_JOB),
        pytest.param("echo -", "echo Exited with exit code 130", LSF_FAILED_JOB),
    ],
)
async def test_exit_codes(tmp_path_factory, bjobs_script, bhist_script, exit_code):
    tmp_path = tmp_path_factory.mktemp("exit_codes")
    mocked_bjobs = tmp_path / "bjobs"
    mocked_bjobs.write_text(f"#!/bin/sh\n{bjobs_script}")
    mocked_bjobs.chmod(mocked_bjobs.stat().st_mode | stat.S_IEXEC)

    mocked_bhist = tmp_path / "bhist"
    mocked_bhist.write_text(f"#!/bin/sh\n{bhist_script}")
    mocked_bhist.chmod(mocked_bhist.stat().st_mode | stat.S_IEXEC)

    driver = LsfDriver(bjobs_cmd=mocked_bjobs, bhist_cmd=mocked_bhist)

    assert await driver._get_exit_code("0") == exit_code


@given(
    jobstate_sequence=st.lists(st.sampled_from(JobState.__args__)),
    exit_code=st.integers(min_value=1, max_value=254),
)
async def test_events_produced_from_jobstate_updates(
    tmp_path_factory, jobstate_sequence: List[str], exit_code: int
):
    tmp_path = tmp_path_factory.mktemp("bjobs_mock")
    mocked_bjobs = tmp_path / "bjobs"
    mocked_bjobs.write_text(f"#!/bin/sh\necho '{exit_code}'")
    mocked_bjobs.chmod(mocked_bjobs.stat().st_mode | stat.S_IEXEC)
    mocked_bhist = tmp_path / "bhist"
    mocked_bhist.write_text("#!/bin/sh\necho 'foo'")
    mocked_bhist.chmod(mocked_bhist.stat().st_mode | stat.S_IEXEC)

    started = any(
        state in jobstate_sequence
        for state in get_type_hints(RunningJob)["job_state"].__args__
    )
    finished_success = any(
        state in jobstate_sequence
        for state in get_type_hints(FinishedJobSuccess)["job_state"].__args__
    )
    finished_failure = any(
        state in jobstate_sequence
        for state in get_type_hints(FinishedJobFailure)["job_state"].__args__
    )

    driver = LsfDriver(bjobs_cmd=mocked_bjobs, bhist_cmd=mocked_bhist)

    async def mocked_submit(self: LsfDriver, iens, name, *_args, **_kwargs):
        """A mocked submit is speedier than going through a command on disk"""
        self._jobs["1"] = JobData(
            iens=iens,
            job_state=QueuedJob(job_state="PEND"),
            submitted_timestamp=time.time(),
        )
        self._iens2jobid[iens] = "1"

    driver.submit = mocked_submit.__get__(driver)
    driver._dump_bhist_job_summary_to_runpath = AsyncMock()
    await driver.submit(0, "_")

    # Replicate the behaviour of multiple calls to poll()
    for statestr in jobstate_sequence:
        jobstate = _parse_jobs_dict({"1": statestr})["1"]
        await driver._process_job_update("1", jobstate)

    events = []
    while not driver.event_queue.empty():
        events.append(await driver.event_queue.get())

    if not started and not finished_success and not finished_failure:
        assert len(events) == 0

        iens, state = driver._jobs["1"].iens, driver._jobs["1"].job_state
        assert iens == 0
        assert isinstance(state, QueuedJob)
    elif started and not finished_success and not finished_failure:
        assert len(events) == 1
        assert events[0] == StartedEvent(iens=0)

        iens, state = driver._jobs["1"].iens, driver._jobs["1"].job_state
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
        assert events[-1] == FinishedEvent(iens=0, returncode=exit_code)
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


@pytest.mark.usefixtures("capturing_bsub")
async def test_submit_with_project_code():
    queue_config_dict = {
        "QUEUE_SYSTEM": "LSF",
        "FORWARD_MODEL": [("FLOW",), ("ECLIPSE",), ("RMS",)],
    }
    queue_config = QueueConfig.from_dict(queue_config_dict)
    driver: LsfDriver = create_driver(queue_config)
    await driver.submit(0, "sleep")
    assert f"-P {queue_config.queue_options.project_code}" in Path(
        "captured_bsub_args"
    ).read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_bsub")
async def test_submit_sets_stdout():
    driver = LsfDriver()
    await driver.submit(0, "sleep", name="myjobname")
    expected_stdout_file = Path(os.getcwd()) / "myjobname.LSF-stdout"
    assert f"-o {expected_stdout_file}" in Path("captured_bsub_args").read_text(
        encoding="utf-8"
    )


@pytest.mark.usefixtures("capturing_bsub")
async def test_submit_sets_stderr():
    driver = LsfDriver()
    await driver.submit(0, "sleep", name="myjobname")
    expected_stdout_file = Path(os.getcwd()) / "myjobname.LSF-stderr"
    assert f"-e {expected_stdout_file}" in Path("captured_bsub_args").read_text(
        encoding="utf-8"
    )


@pytest.mark.usefixtures("capturing_bsub")
async def test_submit_with_resource_requirement():
    driver = LsfDriver(resource_requirement="select[cs && x86_64Linux]")
    await driver.submit(0, "sleep")
    assert "-R select[cs && x86_64Linux]" in Path("captured_bsub_args").read_text(
        encoding="utf-8"
    )
    assert "hname" not in Path("captured_bsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_bsub")
async def test_submit_with_num_cpu():
    driver = LsfDriver()
    await driver.submit(0, "sleep", num_cpu=4)
    assert "-n 4" in Path("captured_bsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_bsub")
async def test_submit_with_realization_memory():
    driver = LsfDriver()
    await driver.submit(0, "sleep", realization_memory=1024**2)
    assert "-R rusage[mem=1]" in Path("captured_bsub_args").read_text(encoding="utf-8")


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
    monkeypatch.chdir(tmp_path)
    bin_path = Path("bin")
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bsub_path = bin_path / "bsub"
    bsub_path.write_text(f"#!/bin/sh\n{bsub_script}")
    bsub_path.chmod(bsub_path.stat().st_mode | stat.S_IEXEC)
    driver = LsfDriver()
    with expectation:
        await driver.submit(0, "sleep")


async def test_faulty_bsub_produces_error_log(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    bin_path = Path("bin")
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")

    _out = "THIS_IS_OUTPUT"
    _err = "THIS_IS_ERROR"
    bsub_script = f"echo {_out} && echo {_err} >&2; exit 1"
    bsub_path = bin_path / "bsub"
    bsub_path.write_text(f"#!/bin/sh\n{bsub_script}")
    bsub_path.chmod(bsub_path.stat().st_mode | stat.S_IEXEC)

    driver = LsfDriver()
    with pytest.raises(RuntimeError):
        await driver.submit(0, "sleep")
    assert (
        f'failed with exit code 1, output: "{_out}", and error: "{_err}"'
        in driver._job_error_message_by_iens[0]
    )


@pytest.mark.timeout(10)
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
            "not submitted properly",
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
            1,
            "",
            "wrong_on_stderr",
            "wrong_on_stderr",
            id="artifical_bkill_stderr_and_returncode_giving_logged_error",
        ),
        pytest.param(
            {"1": "11"},
            "1",
            255,
            "",
            "Job <11>: Job has already finished",
            "",
            id="job_already_finished",
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
    caplog.set_level(logging.INFO)
    monkeypatch.chdir(tmp_path)
    bin_path = Path("bin")
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bkill_path = bin_path / "bkill"
    bkill_path.write_text(
        f"#!/bin/sh\necho '{bkill_stdout}'\n"
        f"echo '{bkill_stderr}' >&2\n"
        f"echo $@ >> 'bkill_args'\n"
        f"exit {bkill_returncode}",
        encoding="utf-8",
    )
    bkill_path.chmod(bkill_path.stat().st_mode | stat.S_IEXEC)

    driver = LsfDriver()
    driver._iens2jobid = mocked_iens2jobid
    driver._sleep_time_between_bkills = 0

    # Needed because we are not submitting anything in this test
    driver._submit_locks[iens_to_kill] = asyncio.Lock()

    await driver.kill(iens_to_kill)

    async def wait_for_sigkill_in_file():
        while True:
            bkill_args_file_content = Path("bkill_args").read_text(encoding="utf-8")

            if "-s SIGKILL" in bkill_args_file_content:
                break
            await asyncio.sleep(0.1)

    if expected_logged_error:
        assert expected_logged_error in caplog.text
    else:
        bkill_args = Path("bkill_args").read_text(encoding="utf-8").strip().split("\n")
        assert f"-s SIGTERM {mocked_iens2jobid[iens_to_kill]}" in bkill_args

        await asyncio.wait_for(wait_for_sigkill_in_file(), timeout=5)


@given(st.text())
def test_parse_bjobs_gives_empty_result_on_random_input(some_text):
    assert parse_bjobs(some_text) == {}


@pytest.mark.parametrize(
    "bjobs_output, expected",
    [
        pytest.param(
            "1^RUN",
            {"1": "RUN"},
            id="basic",
        ),
        pytest.param("1^DONE", {"1": "DONE"}, id="done"),
        pytest.param(
            "1^DONE\n2^RUN",
            {"1": "DONE", "2": "RUN"},
            id="two_jobs",
        ),
    ],
)
def test_parse_bjobs_happy_path(bjobs_output, expected):
    assert parse_bjobs(bjobs_output) == expected


@given(
    st.integers(min_value=1),
    nonempty_string_without_whitespace(),
    st.from_type(JobState),
)
def test_parse_bjobs(job_id, username, job_state):
    assert parse_bjobs(f"{job_id}^{job_state}") == {str(job_id): job_state}


@given(nonempty_string_without_whitespace().filter(lambda x: x not in valid_jobstates))
def test_parse_bjobs_invalid_state_is_ignored(random_state):
    assert parse_bjobs(f"1^{random_state}") == {}


def test_parse_bjobs_invalid_state_is_logged(caplog):
    # (cannot combine caplog with hypothesis)
    parse_bjobs("1^FOO")
    assert "Unknown state FOO" in caplog.text


@pytest.mark.parametrize(
    "bjobs_script, expectation",
    [
        pytest.param(
            "echo '1^DONE'; exit 0",
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
            "echo '1^DONE'; echo 'Job <2> is not found' >&2 ; exit 255",
            # If we have some success and some failures, actual command returns 255
            does_not_raise(),
            id="error_for_irrelevant_job_id",
        ),
        pytest.param(
            "echo '2^DONE'",
            pytest.raises(asyncio.TimeoutError),
            id="wrong-job-id",
        ),
        pytest.param(
            "exit 1",
            pytest.raises(asyncio.TimeoutError),
            id="exit-1",
        ),
        pytest.param(
            "echo '1^DONE'; exit 1",
            # (this is not observed in reality)
            does_not_raise(),
            id="correct_output_but_exitcode_1",
        ),
        pytest.param(
            "echo '1'; exit 0",
            pytest.raises(asyncio.TimeoutError),
            id="unparsable_output",
        ),
    ],
)
async def test_faulty_bjobs(monkeypatch, tmp_path, bjobs_script, expectation):
    monkeypatch.chdir(tmp_path)
    bin_path = Path("bin")
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bsub_path = bin_path / "bsub"
    bsub_path.write_text("#!/bin/sh\necho 'Job <1> is submitted to default queue'")
    bsub_path.chmod(bsub_path.stat().st_mode | stat.S_IEXEC)
    bjobs_path = bin_path / "bjobs"
    bjobs_path.write_text(f"#!/bin/sh\n{bjobs_script}")
    bjobs_path.chmod(bjobs_path.stat().st_mode | stat.S_IEXEC)
    driver = LsfDriver()
    driver._log_bhist_job_summary = AsyncMock()
    with expectation:
        await driver.submit(0, "sleep")
        await asyncio.wait_for(poll(driver, {0}), timeout=1)


@pytest.mark.parametrize(
    ("exit_code, error_msg"),
    [
        (FLAKY_SSH_RETURNCODE, ""),
        (199, "Not recognized"),
    ],
)
async def test_that_bsub_will_retry_and_fail(
    monkeypatch, tmp_path, exit_code, error_msg
):
    monkeypatch.chdir(tmp_path)
    bin_path = Path("bin")
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bsub_path = bin_path / "bsub"
    bsub_path.write_text(f"#!/bin/sh\necho {error_msg} >&2\nexit {exit_code}")
    bsub_path.chmod(bsub_path.stat().st_mode | stat.S_IEXEC)
    driver = LsfDriver()
    driver._bsub_retries = 2
    driver._sleep_time_between_cmd_retries = 0.2
    match_str = (
        f'failed after 2 retries with exit code {exit_code}.*error: "{error_msg if error_msg else "<empty>"}"'
        if exit_code != 199
        else 'failed with exit code 199.*error: "Not recognized"'
    )
    with pytest.raises(RuntimeError, match=match_str):
        await driver.submit(0, "sleep 10")


@pytest.mark.parametrize(
    ("exit_code, error_msg"),
    [
        (0, "void"),
        (FLAKY_SSH_RETURNCODE, ""),
    ],
)
async def test_that_bsub_will_retry_and_succeed(
    monkeypatch, tmp_path, exit_code, error_msg
):
    monkeypatch.chdir(tmp_path)
    bin_path = Path("bin")
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


@pytest.mark.parametrize(
    "resource_requirement, exclude_hosts, realization_memory, expected_string",
    [
        pytest.param(None, None, None, "", id="None input"),
        pytest.param(None, None, 0, "", id="zero_realization_memory_is_None"),
        pytest.param(
            "rusage[mem=50]",
            [],
            None,
            "rusage[mem=50]",
            id="resource_requirement_without_select_and_no_excluded_hosts",
        ),
        pytest.param(
            None,
            [],
            1024 * 1024,
            "rusage[mem=1]",
            id="None_resource_string_with_realization_memory",
        ),
        pytest.param(
            None,
            ["linrgs12-foo", "linrgs13-bar"],
            None,
            "select[hname!='linrgs12-foo' && hname!='linrgs13-bar']",
            id="None_resource_string_with_excluded_hosts",
        ),
        pytest.param(
            "rusage[mem=50]",
            ["linrgs12-foo"],
            None,
            "rusage[mem=50] select[hname!='linrgs12-foo']",
            id="resource_requirement_and_excluded_hosts",
        ),
        pytest.param(
            "rusage[somethingelse=50]",
            [],
            10 * 1024**2,
            "rusage[mem=10,somethingelse=50]",
            id="resource_requirement_without_mem",
        ),
        pytest.param(
            "select[location=='cloud']",
            [],
            10 * 1024**2,
            "select[location=='cloud'] rusage[mem=10]",
            id="select_and_realization_memory",
        ),
        pytest.param(
            "select[location=='cloud']",
            ["linrgs12-foo", "linrgs13-bar"],
            None,
            "select[location=='cloud' && hname!='linrgs12-foo' && hname!='linrgs13-bar']",
            id="existing_select",
        ),
        pytest.param(
            "select[location=='cloud']",
            ["linrgs12-foo", "linrgs13-bar"],
            20 * 1024**3,
            "select[location=='cloud' && hname!='linrgs12-foo' && hname!='linrgs13-bar'] rusage[mem=20480]",
            id="multiple_selects_with_realization_memory",
        ),
        pytest.param(
            None,
            [""],
            None,
            "",
            id="None_resource_requirement_with_empty_string_in_excluded_hosts",
        ),
        pytest.param(
            "rusage[mem=50]",
            [""],
            None,
            "rusage[mem=50]",
            id="resource_requirement_and_empty_string_in_excluded_hosts",
        ),
        pytest.param(
            "select[location=='cloud']",
            [""],
            None,
            "select[location=='cloud']",
            id="select_in_resource_requirement_and_empty_string_in_excluded_hosts",
        ),
        pytest.param(
            "select[location=='cloud'] rusage[mem=7000]",
            [""],
            None,
            "select[location=='cloud'] rusage[mem=7000]",
            id="select_and_rusage_in_resource_requirement_empty_excluded_hosts",
        ),
        pytest.param(
            "select[location=='cloud'] rusage[mem=7000]",
            ["rogue_host"],
            None,
            "select[location=='cloud' && hname!='rogue_host'] rusage[mem=7000]",
            id="select_and_rusage_in_resource_requirement_one_excluded_hosts",
        ),
        pytest.param(
            "rusage[mem=7000] select[location=='cloud']",
            ["rogue_host"],
            None,
            "rusage[mem=7000] select[location=='cloud' && hname!='rogue_host']",
            id="rusage_and_select_resource_requirement_one_excluded_hosts",
        ),
        pytest.param(
            "select[location=='cloud'] rusage[mem=7000]",
            ["rogue_host"],
            6000 * 1024**2,
            "select[location=='cloud' && hname!='rogue_host'] rusage[mem=6000]",
            id="select_and_rusage_in_resource_requirement_one_excluded_hosts",
        ),
    ],
)
def test_build_resource_requirement_string(
    resource_requirement: Optional[str],
    exclude_hosts: List[str],
    realization_memory: Optional[int],
    expected_string: str,
):
    assert (
        build_resource_requirement_string(
            exclude_hosts, realization_memory or 0, resource_requirement or ""
        )
        == expected_string
    )


@pytest.mark.parametrize(
    "bhist_output, expected",
    [
        pytest.param(
            "Summary of time in seconds spent in various states:\n"
            "JOBID  USER  JOB_NAME  PEND    PSUSP  RUN  USUSP  SSUSP  UNKWN  TOTAL\n"
            "1962   user1 *1000000  410650  0      0     0     0      0      410650\n",
            {"1962": {"pending_seconds": 410650, "running_seconds": 0}},
            id="real-world-example-with-one-job",
        ),
        pytest.param(
            "JOBID  USER  JOB_NAME  PEND    PSUSP  RUN  USUSP  SSUSP  UNKWN  TOTAL\n"
            "1962   user1 *echo sl  410650  0      0     0     0      0      410650\n",
            {"1962": {"pending_seconds": 410650, "running_seconds": 0}},
            id="job-name-with-spaces-gives-11-tokens",
        ),
        pytest.param(
            "Summary of time in seconds\n1 x x 3 x 5",
            {"1": {"pending_seconds": 3, "running_seconds": 5}},
            id="shorter-fictous-example",
        ),
        pytest.param(
            "1 x x 3 x 5",
            {"1": {"pending_seconds": 3, "running_seconds": 5}},
            id="minimal-parseable-example",
        ),
        pytest.param(
            "1 x x 3 x 5\n2 x x 4 x 6",
            {
                "1": {"pending_seconds": 3, "running_seconds": 5},
                "2": {"pending_seconds": 4, "running_seconds": 6},
            },
            id="two-jobs-outputted",
        ),
    ],
)
async def test_parse_bhist(bhist_output, expected):
    assert parse_bhist(bhist_output) == expected


empty_states = _parse_jobs_dict({})


@pytest.mark.parametrize(
    "previous_bhist, bhist_output, expected_states",
    [
        pytest.param("", "", empty_states, id="no-input-output"),
        pytest.param("", "1 x x 3 x 5", empty_states, id="no-cache-no-output"),
        pytest.param(
            "1 x x 0 x 0",
            "1 x x 0 x 0",
            _parse_jobs_dict({"1": "DONE"}),
            id="short-job-finished",  # required_cache_age is zero in this test
        ),
        pytest.param(
            "1 x x 1 x 0",
            "1 x x 2 x 0",
            _parse_jobs_dict({"1": "PEND"}),
            id="job-is-pending",
        ),
        pytest.param(
            "1 x x 1 x 0",
            "1 x x 1 x 1",
            _parse_jobs_dict({"1": "RUN"}),
            id="job-is-running",
        ),
        pytest.param(
            "1 x x 1 x 0\n",
            "1 x x 1 x 1\n2 x x 0 x 0",
            _parse_jobs_dict({"1": "RUN"}),
            id="partial_cache",
        ),
        pytest.param(
            "1 x x 1 x 0\n2 x x 0 x 0",
            "1 x x 1 x 1\n2 x x 0 x 0",
            _parse_jobs_dict({"1": "RUN", "2": "DONE"}),
            id="two-jobs",
        ),
        pytest.param(
            "1 x x 1 x 0\n2 x x 0 x 0",
            "2 x x 0 x 0",
            _parse_jobs_dict({"2": "DONE"}),
            id="job-exited-from-cache",
        ),
    ],
)
async def test_poll_once_by_bhist(
    previous_bhist, bhist_output, expected_states, tmp_path
):
    mocked_bhist = tmp_path / "bhist"
    mocked_bhist.write_text(f"#!/bin/sh\necho '{bhist_output}'")
    mocked_bhist.chmod(mocked_bhist.stat().st_mode | stat.S_IEXEC)

    driver = LsfDriver(bhist_cmd=mocked_bhist)
    driver._bhist_cache = parse_bhist(previous_bhist)
    driver._bhist_required_cache_age = 0.0

    before_poll = driver._bhist_required_cache_age
    bhist_states = await driver._poll_once_by_bhist([""])
    # The argument to _poll_once_by_bhist is not relevant as bhist is mocked.

    assert bhist_states == expected_states
    assert driver._bhist_cache_timestamp > before_poll


@pytest.mark.parametrize(
    "required_cache_age, expected_states",
    [
        pytest.param(10, empty_states, id="no_output_for_fresh_cache"),
        pytest.param(
            0,
            _parse_jobs_dict({"1": "DONE"}),
            id="cache_is_old_enough",
        ),
    ],
)
async def test_poll_once_by_bhist_requires_aged_data(
    required_cache_age, expected_states, tmp_path
):
    mocked_bhist = tmp_path / "bhist"
    mocked_bhist.write_text("#!/bin/sh\necho '1 x x 0 x 0'")
    mocked_bhist.chmod(mocked_bhist.stat().st_mode | stat.S_IEXEC)

    driver = LsfDriver(bhist_cmd=mocked_bhist)
    driver._bhist_cache = parse_bhist("1 x x 0 x 0")
    driver._bhist_cache_timestamp = time.time()
    driver._bhist_required_cache_age = required_cache_age
    bhist_states = await driver._poll_once_by_bhist([""])
    # The argument to _poll_once_by_bhist is not relevant as bhist is mocked.

    assert bhist_states == expected_states


@pytest.mark.parametrize(
    "bkill_output",
    [
        "Job <1> is being terminated",
        "Job <1> is being signaled",
    ],
)
async def test_kill_does_not_log_error_on_accepted_bkill_outputs(
    bkill_output, tmp_path, caplog, capsys
):
    bkill_path = tmp_path / "bkill"
    bkill_path.write_text(f"#!/bin/sh\necho '{bkill_output}'; exit 0")
    bkill_path.chmod(bkill_path.stat().st_mode | stat.S_IEXEC)
    driver = LsfDriver(bkill_cmd=bkill_path)

    async def mock_submit(*args, **kwargs):
        driver._iens2jobid[0] = "1"
        driver._submit_locks[0] = asyncio.Lock()

    driver.submit = mock_submit
    await driver.submit(0, "sh", "-c", f"echo test>{tmp_path}/test")
    await driver.kill(0)
    assert "LSF kill failed" not in caplog.text
    assert "LSF kill failed" not in capsys.readouterr().err
    assert "LSF kill failed" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "time_submitted_modifier, expected_result",
    [
        pytest.param(
            -1.0,
            {"1"},
            id="job_submitted_before_deadline",
        ),
        pytest.param(0, set(), id="job_submitted_on_deadline"),
        pytest.param(1.0, set(), id="job_submitted_after_deadline"),
    ],
)
def test_filter_job_ids_on_submission_time(time_submitted_modifier, expected_result):
    submitted_before = time.time()
    job_submitted_timestamp = submitted_before + time_submitted_modifier
    jobs = {
        "1": JobData(
            iens=0,
            job_state=QueuedJob(job_state="PEND"),
            submitted_timestamp=job_submitted_timestamp,
        )
    }
    assert filter_job_ids_on_submission_time(jobs, submitted_before) == expected_result


async def test_kill_before_submit_logs_error(caplog):
    driver = LsfDriver()
    await driver.kill(0)
    assert "ERROR" in caplog.text
    assert "realization 0 has never been submitted" in caplog.text
