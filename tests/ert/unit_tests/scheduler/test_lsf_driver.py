import asyncio
import json
import logging
import os
import random
import re
import stat
import string
import time
from collections.abc import Collection
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from textwrap import dedent
from typing import get_args, get_type_hints
from unittest.mock import AsyncMock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ert.config import QueueConfig
from ert.scheduler import LsfDriver, create_driver
from ert.scheduler.driver import SIGNAL_OFFSET
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
    parse_bjobs_exec_hosts,
)
from tests.ert.utils import poll, wait_until

from .conftest import mock_bin

valid_jobstates: Collection[str] = list(get_args(JobState))


def nonempty_string_without_whitespace():
    return st.text(
        st.characters(whitelist_categories=("Lu", "Ll", "Nd", "P")), min_size=1
    )


def patch_queue_commands(
    monkeypatch,
    tmp_path,
    bsub_script=None,
    bjobs_script=None,
    bhist_script=None,
    bkill_script=None,
):
    monkeypatch.chdir(tmp_path)
    bin_path = Path("bin")
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    if bsub_script:
        bsub_path = bin_path / "bsub"
        bsub_path.write_text("#!/bin/sh\n" + dedent(bsub_script))
        bsub_path.chmod(bsub_path.stat().st_mode | stat.S_IEXEC)
    if bjobs_script:
        bjobs_path = bin_path / "bjobs"
        bjobs_path.write_text("#!/bin/sh\n" + dedent(bjobs_script))
        bjobs_path.chmod(bjobs_path.stat().st_mode | stat.S_IEXEC)
    if bhist_script:
        bhist_path = bin_path / "bhist"
        bhist_path.write_text("#!/bin/sh\n" + dedent(bhist_script))
        bhist_path.chmod(bhist_path.stat().st_mode | stat.S_IEXEC)
    if bkill_script:
        bkill_path = bin_path / "bkill"
        bkill_path.write_text("#!/bin/sh\n" + dedent(bkill_script))
        bkill_path.chmod(bkill_path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def capturing_bsub(monkeypatch, tmp_path):
    patch_queue_commands(
        monkeypatch,
        tmp_path,
        "echo $@ > captured_bsub_args\necho 'Job <1>' is submitted to normal queue",
    )


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
async def test_exit_codes(
    monkeypatch, tmp_path_factory, bjobs_script, bhist_script, exit_code
):
    tmp_path = tmp_path_factory.mktemp("exit_codes")
    patch_queue_commands(
        monkeypatch, tmp_path, bjobs_script=bjobs_script, bhist_script=bhist_script
    )

    driver = LsfDriver()

    assert await driver._get_exit_code("0") == exit_code


@given(
    jobstate_sequence=st.lists(st.sampled_from(JobState.__args__)),
    exit_code=st.integers(min_value=1, max_value=254),
)
async def test_events_produced_from_jobstate_updates(
    tmp_path_factory, jobstate_sequence: list[str], exit_code: int
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
    driver: LsfDriver = create_driver(queue_config.queue_options)
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
    expected_stderr_file = Path(os.getcwd()) / "myjobname.LSF-stderr"
    assert f"-e {expected_stderr_file}" in Path("captured_bsub_args").read_text(
        encoding="utf-8"
    )


@pytest.mark.usefixtures("capturing_bsub")
async def test_submit_with_realization_memory_with_bsub_capture():
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
    patch_queue_commands(monkeypatch, tmp_path, bsub_script=bsub_script)
    driver = LsfDriver()
    driver._max_bsub_attempts = 1
    with expectation:
        await driver.submit(0, "sleep")


async def test_faulty_bsub_produces_error_log(monkeypatch, tmp_path):
    out = "THIS_IS_OUTPUT"
    err = "THIS_IS_ERROR"
    patch_queue_commands(
        monkeypatch, tmp_path, bsub_script=f"echo {out} && echo {err} >&2; exit 1"
    )
    monkeypatch.chdir(tmp_path)
    driver = LsfDriver()
    with pytest.raises(RuntimeError):
        await driver.submit(0, "sleep")
    assert (
        f'failed with exit code 1, output: "{out}", and error: "{err}"'
        in driver._job_error_message_by_iens[0]
    )


@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "mocked_iens2jobid, iens_to_kill, "
    "bkill_returncode, bkill_stdout, bkill_stderr, expected_logged_error",
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
            id="artificial_bkill_stdout_giving_logged_error",
        ),
        pytest.param(
            {"1": "11"},
            "1",
            1,
            "",
            "wrong_on_stderr",
            "wrong_on_stderr",
            id="artificial_bkill_stderr_and_returncode_giving_logged_error",
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
    patch_queue_commands(
        monkeypatch,
        tmp_path,
        bkill_script=f"echo '{bkill_stdout}'\n"
        f"echo '{bkill_stderr}' >&2\n"
        f"echo $@ >> 'bkill_args'\n"
        f"exit {bkill_returncode}",
    )

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
            "1^RUN^-",
            {"1": "RUN"},
            id="basic",
        ),
        pytest.param("1^DONE^-", {"1": "DONE"}, id="done"),
        pytest.param(
            "1^DONE^-\n2^RUN^-",
            {"1": "DONE", "2": "RUN"},
            id="two_jobs",
        ),
    ],
)
def test_parse_bjobs_happy_path(bjobs_output, expected):
    assert parse_bjobs(bjobs_output) == expected


@pytest.mark.parametrize(
    "bjobs_output, expected",
    [
        pytest.param(
            "1^RUN^abc-comp01",
            {"1": "abc-comp01"},
            id="one_host",
        ),
        pytest.param(
            "1^DONE^abc-comp02\n2^RUN^-",
            {"1": "abc-comp02", "2": "-"},
            id="two_hosts_output",
        ),
    ],
)
def test_parse_bjobs_exec_hosts_happy_path(bjobs_output, expected):
    assert parse_bjobs_exec_hosts(bjobs_output) == expected


@given(
    st.integers(min_value=1),
    st.from_type(JobState),
)
def test_parse_bjobs(job_id, job_state):
    assert parse_bjobs(f"{job_id}^{job_state}^-") == {str(job_id): job_state}


@given(
    st.integers(min_value=1),
    st.from_type(JobState),
    nonempty_string_without_whitespace(),
)
def test_parse_bjobs_exec_host(job_id, job_state, exec_host):
    assert parse_bjobs_exec_hosts(f"{job_id}^{job_state}^{exec_host}") == {
        str(job_id): exec_host
    }


@given(nonempty_string_without_whitespace().filter(lambda x: x not in valid_jobstates))
def test_parse_bjobs_invalid_state_is_ignored(random_state):
    assert parse_bjobs(f"1^{random_state}") == {}


def test_parse_bjobs_invalid_state_is_logged(caplog):
    # (cannot combine caplog with hypothesis)
    parse_bjobs("1^FOO^-")
    assert "Unknown state FOO" in caplog.text


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "bjobs_script, expectation",
    [
        pytest.param(
            "echo '1^DONE^-'; exit 0",
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
            "echo '1^DONE^-'; echo 'Job <2> is not found' >&2 ; exit 255",
            # If we have some success and some failures, actual command returns 255
            does_not_raise(),
            id="error_for_irrelevant_job_id",
        ),
        pytest.param(
            "echo '2^DONE^-'",
            pytest.raises(asyncio.TimeoutError),
            id="wrong-job-id",
        ),
        pytest.param(
            "exit 1",
            pytest.raises(asyncio.TimeoutError),
            id="exit-1",
        ),
        pytest.param(
            "echo '1^DONE^-'; exit 1",
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
    patch_queue_commands(
        monkeypatch,
        tmp_path,
        bsub_script="echo 'Job <1> is submitted to default queue'",
        bjobs_script=bjobs_script,
    )
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
    patch_queue_commands(
        monkeypatch, tmp_path, bsub_script=f"echo {error_msg} >&2\nexit {exit_code}"
    )
    driver = LsfDriver()
    driver._max_bsub_attempts = 2
    driver._sleep_time_between_cmd_retries = 0.0
    match_str = (
        f"failed after 2 attempts with exit code {exit_code}.*"
        f'error: "{error_msg or "<empty>"}"'
        if exit_code != 199
        else 'failed with exit code 199.*error: "Not recognized"'
    )
    with pytest.raises(RuntimeError, match=match_str):
        await driver.submit(0, "sleep 10")


@pytest.mark.parametrize(
    ("exit_code, error_msg"),
    [
        # All these have been manually obtained on the command line by
        # perturbing the command arguments to bsub:
        (255, "No such queue. Job not submitted"),
        (255, "Too many processors requested. Job not submitted."),
        (255, 'Error near "select" : duplicate section. Job not submitted.'),
        (
            255,
            "Error in select section: Expected number, string, "
            'name, or "(" but found end of section. Job not submitted.',
        ),
        (
            255,
            "Error with <select[rhel < 8 && cs & x86_64Linux] rusage[mem=50]>:"
            " '&' cannot be used in the resource requirement section. "
            "Job not submitted.",
        ),
        (255, "Error in rusage section. Job not submitted."),
    ],
)
async def test_that_bsub_will_fail_without_retries(
    monkeypatch, tmp_path, exit_code, error_msg
):
    patch_queue_commands(
        monkeypatch,
        tmp_path,
        f'echo . >> bsubcalls\necho "{error_msg}" >&2\nexit {exit_code}',
    )
    driver = LsfDriver()
    with pytest.raises(RuntimeError):
        await driver.submit(0, "sleep 10")
    assert len(Path("bsubcalls").read_text(encoding="utf-8").strip()) == 1


@pytest.mark.parametrize(
    ("exit_code, error_msg"),
    [
        (FLAKY_SSH_RETURNCODE, ""),
        (FLAKY_SSH_RETURNCODE, "Request from non-LSF host rejected"),
    ],
)
async def test_that_bsub_will_retry_and_succeed(
    monkeypatch, tmp_path, exit_code, error_msg
):
    patch_queue_commands(
        monkeypatch,
        tmp_path,
        f"""
            TRY_FILE="{tmp_path}/script_try"
            if [ -f "$TRY_FILE" ]; then
                echo "Job <1> is submitted to normal queue"
                exit 0
            else
                echo "TRIED" > $TRY_FILE
                echo "{error_msg}" >&2
                exit {exit_code}
            fi
            """,
    )
    driver = LsfDriver()
    driver._max_bsub_attempts = 2
    driver._sleep_time_between_cmd_retries = 0.0
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
            (
                "select[location=='cloud' && hname!='linrgs12-foo' "
                "&& hname!='linrgs13-bar']"
            ),
            id="existing_select",
        ),
        pytest.param(
            "select[location=='cloud']",
            ["linrgs12-foo", "linrgs13-bar"],
            20 * 1024**3,
            (
                "select[location=='cloud' && hname!='linrgs12-foo' && "
                "hname!='linrgs13-bar'] rusage[mem=20480]"
            ),
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
    resource_requirement: str | None,
    exclude_hosts: list[str],
    realization_memory: int | None,
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
    previous_bhist, bhist_output, expected_states, tmp_path, monkeypatch
):
    patch_queue_commands(monkeypatch, tmp_path, bhist_script=f"echo '{bhist_output}'")

    driver = LsfDriver()
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
    required_cache_age, expected_states, tmp_path, monkeypatch
):
    patch_queue_commands(monkeypatch, tmp_path, bhist_script="echo '1 x x 0 x 0'")

    driver = LsfDriver()
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
    bkill_output, tmp_path, caplog, capsys, monkeypatch
):
    patch_queue_commands(
        monkeypatch, tmp_path, bkill_script=f"echo '{bkill_output}'; exit 0"
    )
    driver = LsfDriver()

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
    caplog.set_level(logging.DEBUG)
    driver = LsfDriver()
    await driver.kill(0)
    assert "DEBUG" in caplog.text
    assert "LSF kill was not run, realization 0 has never been submitted" in caplog.text


@pytest.fixture(autouse=True)
def mock_lsf(pytestconfig, monkeypatch, tmp_path):
    if pytestconfig.getoption("lsf"):
        # User provided --lsf, which means we should use the actual LSF
        # cluster without mocking anything.""
        return
    mock_bin(monkeypatch, tmp_path)


@pytest.fixture
def not_found_bjobs(monkeypatch, tmp_path):
    """This creates a bjobs command that will always claim a job
    does not exist, mimicking a job that has 'fallen out of the bjobs cache'."""
    patch_queue_commands(
        monkeypatch, tmp_path, bjobs_script='echo "Job <$1> is not found"'
    )


@pytest.mark.integration_test
async def test_bjobs_exec_host_logs_only_once(use_tmpdir, job_name, caplog):
    caplog.set_level(logging.INFO)
    driver = LsfDriver()
    await driver.submit(0, "sh", "-c", "sleep 1", name=job_name)

    job_id = next(iter(driver._jobs.keys()))
    driver.update_and_log_exec_hosts({job_id: "COMP-01"})
    driver.update_and_log_exec_hosts({job_id: "COMP-02"})

    await poll(driver, {0})
    assert caplog.text.count("was assigned to host:") == 1


@pytest.mark.integration_test
async def test_lsf_stdout_file(use_tmpdir, job_name):
    driver = LsfDriver()
    await driver.submit(0, "sh", "-c", "echo yay", name=job_name)
    await poll(driver, {0})
    lsf_stdout = Path(f"{job_name}.LSF-stdout").read_text(encoding="utf-8")
    assert Path(f"{job_name}.LSF-stdout").exists(), (
        "LSF system did not write output file"
    )

    assert "Sender: " in lsf_stdout, "LSF stdout should always start with 'Sender:'"
    assert "The output (if any) follows:" in lsf_stdout
    assert "yay" in lsf_stdout


@pytest.mark.integration_test
async def test_lsf_dumps_stderr_to_file(use_tmpdir, job_name):
    driver = LsfDriver()
    failure_message = "failURE"
    await driver.submit(0, "sh", "-c", f"echo {failure_message} >&2", name=job_name)
    await poll(driver, {0})
    assert Path(f"{job_name}.LSF-stderr").exists(), (
        "LSF system did not write stderr file"
    )

    assert (
        Path(f"{job_name}.LSF-stderr").read_text(encoding="utf-8").strip()
        == failure_message
    )


def generate_random_text(size):
    letters = string.ascii_letters
    return "".join(random.choice(letters) for i in range(size))


@pytest.mark.integration_test
@pytest.mark.parametrize("tail_chars_to_read", [(5), (50), (500), (700)])
async def test_lsf_can_retrieve_stdout_and_stderr(
    use_tmpdir, job_name, tail_chars_to_read
):
    driver = LsfDriver()
    num_written_characters = 600
    out = generate_random_text(num_written_characters)
    err = generate_random_text(num_written_characters)
    await driver.submit(0, "sh", "-c", f"echo {out} && echo {err} >&2", name=job_name)
    await poll(driver, {0})
    message = driver.read_stdout_and_stderr_files(
        runpath=".",
        job_name=job_name,
        num_characters_to_read_from_end=tail_chars_to_read,
    )

    stderr_txt = Path(f"{job_name}.LSF-stderr").read_text(encoding="utf-8").strip()
    stdout_txt = Path(f"{job_name}.LSF-stdout").read_text(encoding="utf-8").strip()

    assert stderr_txt[-min(tail_chars_to_read, num_written_characters) + 2 :] in message
    assert stdout_txt[-min(tail_chars_to_read, num_written_characters) + 2 :] in message


@pytest.mark.integration_test
async def test_lsf_cannot_retrieve_stdout_and_stderr(use_tmpdir, job_name):
    driver = LsfDriver()
    num_written_characters = 600
    out = generate_random_text(num_written_characters)
    err = generate_random_text(num_written_characters)
    await driver.submit(0, "sh", "-c", f"echo {out} && echo {err} >&2", name=job_name)
    await poll(driver, {0})
    # let's remove the output files
    os.remove(job_name + ".LSF-stderr")
    os.remove(job_name + ".LSF-stdout")
    message = driver.read_stdout_and_stderr_files(
        runpath=".",
        job_name=job_name,
        num_characters_to_read_from_end=1,
    )
    assert "LSF-stderr:\nNo output file" in message
    assert "LSF-stdout:\nNo output file" in message


@pytest.mark.integration_test
@pytest.mark.parametrize("explicit_runpath", [(True), (False)])
async def test_lsf_info_file_in_runpath(
    explicit_runpath, tmp_path, job_name, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    driver = LsfDriver()
    (tmp_path / "some_runpath").mkdir()
    effective_runpath = tmp_path / "some_runpath" if explicit_runpath else tmp_path
    await driver.submit(
        0,
        "sh",
        "-c",
        "exit 0",
        runpath=tmp_path / "some_runpath" if explicit_runpath else None,
        name=job_name,
    )

    await poll(driver, {0})

    effective_runpath = tmp_path / "some_runpath" if explicit_runpath else tmp_path
    assert json.loads(
        (effective_runpath / "lsf_info.json").read_text(encoding="utf-8")
    ).keys() == {"job_id"}


@pytest.mark.integration_test
async def test_submit_to_named_queue(tmp_path, caplog, job_name, monkeypatch):
    """If the environment variable _ERT_TEST_ALTERNATIVE_QUEUE is defined
    a job will be attempted submitted to that queue.

    As Ert does not keep track of which queue a job is executed in, we can only
    test for success for the job."""
    monkeypatch.chdir(tmp_path)
    driver = LsfDriver(queue_name=os.getenv("_ERT_TESTS_ALTERNATIVE_QUEUE"))
    await driver.submit(0, "sh", "-c", f"echo test > {tmp_path}/test", name=job_name)
    await poll(driver, {0})

    assert (tmp_path / "test").read_text(encoding="utf-8") == "test\n"


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
async def test_submit_with_resource_requirement(job_name):
    resource_requirement = "select[cs && x86_64Linux]"
    driver = LsfDriver(resource_requirement=resource_requirement)
    await driver.submit(0, "sh", "-c", "echo test>test", name=job_name)
    await poll(driver, {0})

    assert Path("test").read_text(encoding="utf-8") == "test\n"


@pytest.mark.usefixtures("capturing_bsub")
async def test_submit_with_resource_requirement_with_bsub_capture():
    driver = LsfDriver(resource_requirement="select[cs && x86_64Linux]")
    await driver.submit(0, "sleep")
    assert "-R select[cs && x86_64Linux]" in Path("captured_bsub_args").read_text(
        encoding="utf-8"
    )
    assert "hname" not in Path("captured_bsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_bsub")
async def test_empty_job_name():
    driver = LsfDriver()
    await driver.submit(0, "/bin/sleep")
    assert " -J sleep " in Path("captured_bsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("use_tmpdir")
async def test_submit_with_num_cpu(pytestconfig, job_name):
    if not pytestconfig.getoption("lsf"):
        return

    num_cpu = 2
    driver = LsfDriver()
    await driver.submit(0, "sh", "-c", "echo test>test", name=job_name, num_cpu=num_cpu)
    job_id = driver._iens2jobid[0]
    await poll(driver, {0})

    process = await asyncio.create_subprocess_exec(
        "bhist",
        "-l",
        job_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    stdout_no_whitespaces = re.sub(r"\s+", "", stdout.decode())
    matches = re.search(r".*([0-9]+)ProcessorsRequested.*", stdout_no_whitespaces)
    assert matches and matches[1] == str(num_cpu), (
        f"Could not verify processor allocation from stdout: {stdout}, stderr: {stderr}"
    )

    assert Path("test").read_text(encoding="utf-8") == "test\n"


@pytest.mark.usefixtures("capturing_bsub")
async def test_submit_with_num_cpu_with_bsub_capture():
    driver = LsfDriver()
    await driver.submit(0, "sleep", num_cpu=4)
    assert "-n 4" in Path("captured_bsub_args").read_text(encoding="utf-8")


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
async def test_submit_with_realization_memory(pytestconfig, job_name):
    if not pytestconfig.getoption("lsf"):
        pytest.skip("Mocked LSF driver does not provide bhist")

    realization_memory_bytes = 1024 * 1024
    driver = LsfDriver()
    await driver.submit(
        0,
        "sh",
        "-c",
        "echo test>test",
        name=job_name,
        realization_memory=realization_memory_bytes,
    )
    job_id = driver._iens2jobid[0]
    await poll(driver, {0})

    process = await asyncio.create_subprocess_exec(
        "bhist",
        "-l",
        job_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    assert "rusage[mem=1]" in stdout.decode(encoding="utf-8")

    assert Path("test").read_text(encoding="utf-8") == "test\n"


@pytest.mark.integration_test
async def test_polling_bhist_fallback(not_found_bjobs, caplog, job_name):
    caplog.set_level(logging.DEBUG)
    driver = LsfDriver()
    Path("mock_jobs").mkdir()
    Path("mock_jobs/pendingtimemillis").write_text("100", encoding="utf-8")
    driver._poll_period = 0.01

    bhist_called = False
    original_bhist_method = driver._poll_once_by_bhist

    def mock_poll_once_by_bhist(*args, **kwargs):
        nonlocal bhist_called
        bhist_called = True
        return original_bhist_method(*args, **kwargs)

    driver._poll_once_by_bhist = mock_poll_once_by_bhist

    await driver.submit(0, "sh", "-c", "sleep 1", name=job_name)
    job_id = next(iter(driver._iens2jobid.values()))
    await poll(driver, {0})
    assert "bhist is used" in caplog.text
    assert bhist_called
    assert driver._bhist_cache
    assert job_id in driver._bhist_cache


async def test_no_exception_when_no_access_to_bjobs_executable(
    not_found_bjobs, caplog, job_name
):
    """The intent of this test is to ensure the driver will not
    go down if the filesystem is temporarily flaky."""
    driver = LsfDriver()
    driver._poll_period = 0.01
    Path("bin/bjobs").chmod(0o000)  # Modify the bjobs from the fixture
    await driver.submit(0, "sh", "-c", "echo", name=job_name)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(driver.poll(), timeout=0.1)
    assert "Permission denied" in caplog.text


async def test_jobname_with_spaces(use_tmpdir, pytestconfig):
    if not pytestconfig.getoption("lsf"):
        pytest.skip("Mocked LSF driver does not support spaces")
    driver = LsfDriver()
    await driver.submit(0, "sh", "-c", "sleep 1", name="I have spaces")
    await poll(driver, {0})


@pytest.mark.integration_test
async def test_that_kill_before_submit_is_finished_works(tmp_path, monkeypatch, caplog):
    """This test asserts that it is possible to issue a kill command
    to a realization right after it has been submitted (as in driver.submit()).

    The bug intended to catch is if the driver gives up on killing before submission
    is not done yet, it is important not to let the realization through in that
    scenario.
    """
    monkeypatch.chdir(tmp_path)

    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bsub_path = bin_path / "slow_bsub"
    bsub_path.write_text(
        "#!/bin/sh\nsleep 0.1\nbsub $@",
        encoding="utf-8",
    )
    bsub_path.chmod(bsub_path.stat().st_mode | stat.S_IEXEC)

    caplog.set_level(logging.DEBUG)
    driver = LsfDriver(bsub_cmd="slow_bsub")

    job_path = bin_path / "job.sh"
    job_path.write_text(
        dedent(
            f"""\
            #!/bin/bash
            do_stop=0

            function handle()
            {{
                echo "killed" > {tmp_path}/was_killed
                do_stop=1
                exit 15
            }}
            trap handle SIGTERM
            touch {tmp_path}/trap_handle_installed
            while [[ $do_stop == 0 ]]
            do
                sleep 0.1
            done
            """
        ),
        encoding="utf-8",
    )
    job_path.chmod(job_path.stat().st_mode | stat.S_IEXEC)

    # Allow submit and kill to be interleaved by asyncio by issuing
    # submit() in its own asyncio Task:
    asyncio.create_task(
        driver.submit(
            0,
            str(job_path),
        )
    )
    await asyncio.sleep(0.01)  # Allow submit task to start executing
    # This will wait until the submit is done and then kill
    await driver.kill(0)

    async def finished(iens: int, returncode: int):
        SIGTERM = 15
        assert iens == 0
        # If the kill is issued before the job really starts, you will not
        # get SIGTERM but rather LSF_FAILED_JOB. Whether SIGNAL_OFFSET is
        # added or not depends on various shell configurations and is a
        # detail we do not want to track.
        assert returncode in {SIGTERM, SIGNAL_OFFSET + SIGTERM, LSF_FAILED_JOB}

    await poll(driver, {0}, finished=finished)
    assert "ERROR" not in str(caplog.text)

    # Normally the job script in this test should never get a chance to start,
    # but if the system is loaded, the script will start before the kill command is
    # processed. If the script starts, we assert that it was killed in
    # a controlled fashion:
    if (tmp_path / "trap_handle_installed").exists():
        wait_until((tmp_path / "was_killed").exists, timeout=4)
