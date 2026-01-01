import asyncio
import contextlib
import json
import logging
import os
import shlex
import stat
from functools import partial
from pathlib import Path
from textwrap import dedent

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ert.cli.main import ErtCliError
from ert.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert.mode_definitions import ENSEMBLE_EXPERIMENT_MODE
from ert.scheduler.openpbs_driver import (
    JOB_STATES,
    QDEL_JOB_HAS_FINISHED,
    QDEL_REQUEST_INVALID,
    QSUB_CONNECTION_REFUSED,
    QSUB_INVALID_CREDENTIAL,
    QSUB_PREMATURE_END_OF_MESSAGE,
    FinishedEvent,
    FinishedJob,
    OpenPBSDriver,
    QueuedJob,
    RunningJob,
    StartedEvent,
    _create_job_class,
    _parse_jobs_dict,
)
from ert.scheduler.scheduler import Scheduler
from tests.ert.ui_tests.cli.run_cli import run_cli
from tests.ert.utils import poll

from .conftest import mock_bin

pytestmark = pytest.mark.xdist_group("openpbs")


QSTAT_HEADER = (
    "Job id                         Name            User             Time Use S Queue\n"
    "-----------------------------  --------------- ---------------  -------- - ---------------\n"  # noqa: E501
)
QSTAT_HEADER_FORMAT = "%-30s %-15s %-15s %-8s %-1s %-5s"


@given(st.lists(st.sampled_from(JOB_STATES)))
async def test_events_produced_from_jobstate_updates(jobstate_sequence: list[str]):
    # Determine what to expect from the sequence:
    started = False
    finished = False
    if "R" in jobstate_sequence:
        started = True
    if "F" in jobstate_sequence or "E" in jobstate_sequence:
        finished = True

    driver = OpenPBSDriver()

    async def mocked_submit(self, iens, *_args, **_kwargs):
        """A mocked submit is speedier than going through a command on disk"""
        self._jobs["1"] = (iens, QueuedJob())
        self._iens2jobid[iens] = "1"
        self._non_finished_job_ids.add("1")

    driver.submit = mocked_submit.__get__(driver)
    await driver.submit(0, "_")

    # Replicate the behaviour of multiple calls to poll()
    for statestr in jobstate_sequence:
        jobstate = _parse_jobs_dict({"1": {"job_state": statestr, "Exit_status": 0}})[
            "1"
        ]
        if statestr in {"E", "F"} and "1" in driver._non_finished_job_ids:
            driver._non_finished_job_ids.remove("1")
            driver._finished_job_ids.add("1")
        await driver._process_job_update("1", jobstate)

    events = []
    while not driver.event_queue.empty():
        events.append(await driver.event_queue.get())

    if started is False and finished is False:
        assert len(events) == 0

        iens, state = driver._jobs["1"]
        assert iens == 0
        assert isinstance(state, QueuedJob)
    elif started is True and finished is False:
        assert len(events) == 1
        assert events[0] == StartedEvent(iens=0)

        iens, state = driver._jobs["1"]
        assert iens == 0
        assert isinstance(state, RunningJob)
    elif started is True and finished is True:
        assert len(events) <= 2  # The StartedEvent is not required
        assert events[-1] == FinishedEvent(iens=0, returncode=0)
        assert "1" not in driver._jobs


words = st.text(
    min_size=0,
    max_size=8,
    alphabet=st.characters(min_codepoint=ord("A"), max_codepoint=ord("Z")),
)


@pytest.fixture
def capturing_qsub(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    qsub_path = bin_path / "qsub"
    qsub_path.write_text(
        "#!/bin/sh\necho $@ > captured_qsub_args; echo '1'", encoding="utf-8"
    )
    qsub_path.chmod(qsub_path.stat().st_mode | stat.S_IEXEC)


def parse_resource_string(qsub_args: str) -> dict[str, str]:
    resources = {}

    args = shlex.split(qsub_args)
    for dash_l_index in [idx for idx, arg in enumerate(args) if arg == "-l"]:
        resource_string = args[dash_l_index + 1]

        for key_value in resource_string.split(":"):
            if "=" in key_value:
                key, value = key_value.split("=")
                resources[key] = value
            else:
                resources[key_value] = "_present_"
    return resources


@pytest.mark.usefixtures("capturing_qsub")
async def test_no_default_realization_memory():
    driver = OpenPBSDriver()
    await driver.submit(0, "sleep")
    assert " -l " not in Path("captured_qsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_qsub")
async def test_realization_memory_not_negative():
    # Validation will happen during config parsing
    driver = OpenPBSDriver()
    await driver.submit(0, "sleep", realization_memory=-1)
    assert " -l " not in Path("captured_qsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_qsub")
async def test_job_name():
    driver = OpenPBSDriver()
    await driver.submit(0, "sleep", name="sleepy")
    assert " -Nsleepy " in Path("captured_qsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_qsub")
async def test_empty_job_name():
    driver = OpenPBSDriver()
    await driver.submit(0, "/bin/sleep")
    assert " -Nsleep " in Path("captured_qsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_qsub")
async def test_job_name_with_prefix():
    driver = OpenPBSDriver(job_prefix="pre_")
    await driver.submit(0, "sleep", name="sleepy")
    assert " -Npre_sleepy " in Path("captured_qsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_qsub")
async def test_cluster_label():
    driver = OpenPBSDriver(cluster_label="foobar")
    await driver.submit(0, "sleep")
    assert "-l foobar" in Path("captured_qsub_args").read_text(encoding="utf-8")


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "qstat_script, started_expected",
    [
        pytest.param(
            f"echo '{QSTAT_HEADER}';printf '{QSTAT_HEADER_FORMAT}' "
            "1 foo someuser 0 R normal; exit 0",
            True,
            id="all-good",
        ),
        pytest.param(
            (
                f"echo '{QSTAT_HEADER}'; "
                f"printf '{QSTAT_HEADER_FORMAT}' 1 foo someuser 0 R normal"
            ),
            True,
            id="all-good-properly-formatted",
        ),
        pytest.param(
            "echo ''; exit 0",
            False,
            id="empty_cluster",
        ),
        pytest.param(
            "echo 'qstat: Unknown Job Id 1' >&2; exit 1",
            False,
            id="empty_cluster_specific_id",
        ),
        pytest.param(
            f"printf '{QSTAT_HEADER_FORMAT}' 1 foo someuser 0 Z normal",
            False,
            id="unknown_jobstate_token_from_pbs",  # Never observed
        ),
        pytest.param(
            f"echo '{QSTAT_HEADER}'; printf '{QSTAT_HEADER_FORMAT}' "
            "1 foo someuser 0 R normal; "
            "echo 'qstat: Unknown Job Id 2' >&2 ; exit 153",
            # If we have some success and some failures, actual command returns 153
            True,
            id="error_for_irrelevant_job_id",
        ),
        pytest.param(
            f"echo '{QSTAT_HEADER}'; printf '{QSTAT_HEADER_FORMAT}' "
            "2 foo someuser 0 R normal",
            False,
            id="wrong-job-id",
        ),
        pytest.param(
            "exit 1",
            False,
            id="exit-1",
        ),
        pytest.param(
            f"echo '{QSTAT_HEADER}1 foo'; exit 0",
            False,
            id="unparsable_output",  # Never observed
        ),
    ],
)
async def test_faulty_qstat(monkeypatch, tmp_path, qstat_script, started_expected):
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    qsub_path = bin_path / "qsub"
    qsub_path.write_text("#!/bin/sh\necho '1'")
    qsub_path.chmod(qsub_path.stat().st_mode | stat.S_IEXEC)
    qstat_path = bin_path / "qstat"
    qstat_path.write_text(f"#!/bin/sh\n{qstat_script}")
    qstat_path.chmod(qstat_path.stat().st_mode | stat.S_IEXEC)
    driver = OpenPBSDriver()
    await driver.submit(0, "sleep")

    was_started = False

    async def started(realizations: list[int]):
        nonlocal was_started
        if realizations[0] == 0:
            was_started = True

    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(poll(driver, expected=set(), started=started), timeout=1)

    assert was_started == started_expected


@given(st.integers(), st.integers(min_value=1), words)
@settings(max_examples=10)
@pytest.mark.usefixtures("capturing_qsub")
async def test_full_resource_string(realization_memory, num_cpu, cluster_label):
    driver = OpenPBSDriver(
        cluster_label=cluster_label or None,
    )
    await driver.submit(
        0, "sleep", num_cpu=num_cpu, realization_memory=realization_memory
    )
    resources = parse_resource_string(
        Path("captured_qsub_args").read_text(encoding="utf-8")
    )
    assert resources.get("mem", "") == (
        f"{realization_memory // 1024**2}mb" if realization_memory > 0 else ""
    )
    assert resources.get("select", "1") == "1"
    assert resources.get("ncpus", "1") == str(num_cpu)

    if cluster_label:
        # cluster_label is not a key-value thing in the resource list,
        # the parser in this test handles that specially
        assert resources.get(cluster_label) == "_present_"

    assert len(resources) == sum(
        [
            realization_memory > 0,
            num_cpu > 1,
            bool(cluster_label),
        ]
    ), "Unknown or missing resources in resource string"


@pytest.mark.parametrize(
    ("exit_code, error_msg"),
    [
        (QSUB_INVALID_CREDENTIAL, "Invalid credential"),
        (QSUB_PREMATURE_END_OF_MESSAGE, "Premature end of message"),
        (QSUB_CONNECTION_REFUSED, "Connection refused"),
        (199, "Not recognized"),
    ],
)
async def test_that_qsub_will_retry_and_fail(
    monkeypatch, tmp_path, exit_code, error_msg
):
    monkeypatch.chdir(tmp_path)
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    qsub_path = bin_path / "qsub"
    qsub_path.write_text(f"#!/bin/sh\necho {error_msg} >&2\nexit {exit_code}")
    qsub_path.chmod(qsub_path.stat().st_mode | stat.S_IEXEC)
    driver = OpenPBSDriver()
    driver._max_pbs_cmd_attempts = 2
    driver._sleep_time_between_cmd_retries = 0.01
    match_str = (
        f'failed after 2 attempts with exit code {exit_code}.*error: "{error_msg}"'
        if exit_code != 199
        else 'failed with exit code 199.*error: "Not recognized"'
    )
    with pytest.raises(RuntimeError, match=match_str):
        await driver.submit(0, "sleep 10")


@pytest.mark.parametrize(
    ("exit_code, error_msg"),
    [
        (QSUB_INVALID_CREDENTIAL, "Invalid credential"),
        (QSUB_PREMATURE_END_OF_MESSAGE, "Premature end of message"),
        (QSUB_CONNECTION_REFUSED, "Connection refused"),
    ],
)
async def test_that_qsub_will_retry_and_succeed(
    monkeypatch, tmp_path, exit_code, error_msg
):
    monkeypatch.chdir(tmp_path)
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    qsub_path = bin_path / "qsub"
    qsub_path.write_text(
        "#!/bin/sh"
        + dedent(
            f"""
            TRY_FILE="{bin_path}/script_try"
            if [ -f "$TRY_FILE" ]; then
                echo "SUCCESS"
                exit 0
            else
                echo "TRIED" > $TRY_FILE
                echo "{error_msg}" >&2
                exit {exit_code}
            fi
            """
        )
    )
    qsub_path.chmod(qsub_path.stat().st_mode | stat.S_IEXEC)
    driver = OpenPBSDriver()
    driver._poll_period = 0.01
    driver._max_pbs_cmd_attempts = 2
    driver._sleep_time_between_cmd_retries = 0.01
    await driver.submit(0, "sleep 10")


@pytest.mark.parametrize(
    ("exit_code, error_msg"),
    [
        (QDEL_JOB_HAS_FINISHED, "Job has finished\nJob has finished"),
        (QDEL_REQUEST_INVALID, "Request invalid for state of job"),
    ],
)
async def test_that_qdel_will_retry_and_succeed(
    monkeypatch, tmp_path, exit_code, error_msg
):
    monkeypatch.chdir(tmp_path)
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    qdel_path = bin_path / "qdel"
    qdel_path.write_text(
        "#!/bin/sh"
        + dedent(
            f"""
            TRY_FILE="{bin_path}/script_try"
            if [ -f "$TRY_FILE" ]; then
                echo "qdel executed" > "{bin_path}/qdel_output"
                exit 0
            else
                echo "TRIED" > $TRY_FILE
                echo "{error_msg}" > "{bin_path}/qdel_error"
                exit {exit_code}
            fi
            """
        )
    )
    qdel_path.chmod(qdel_path.stat().st_mode | stat.S_IEXEC)
    driver = OpenPBSDriver()
    driver._max_pbs_cmd_attempts = 2
    driver._sleep_time_between_cmd_retries = 0.01
    driver._iens2jobid[0] = str(111)
    driver._iens2jobid[1] = str(222)
    await driver.kill([0, 1])
    assert "TRIED" in (bin_path / "script_try").read_text()
    if exit_code == QDEL_JOB_HAS_FINISHED:
        # the job has been already qdel-ed so no need to retry
        assert not (bin_path / "qdel_output").exists()
    else:
        assert "qdel executed" in (bin_path / "qdel_output").read_text()
    assert error_msg in (bin_path / "qdel_error").read_text()


@pytest.mark.usefixtures("capturing_qsub")
@pytest.mark.parametrize("value", [True, False])
async def test_keep_qsub_output(value: bool):
    driver = OpenPBSDriver(keep_qsub_output=value)
    await driver.submit(0, "sleep")
    if value:
        assert "dev/null" not in Path("captured_qsub_args").read_text(encoding="utf-8")
    else:
        assert " -o /dev/null -e /dev/null" in Path("captured_qsub_args").read_text(
            encoding="utf-8"
        )


@pytest.fixture
def create_mock_flaky_qstat(monkeypatch, tmp_path):
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.chdir(bin_path)
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    yield _mock_flaky_qstat


def _mock_flaky_qstat(error_message_to_output: str):
    qsub_path = Path("qsub")
    qsub_path.write_text("#!/bin/sh\necho '1'", encoding="utf-8")
    qsub_path.chmod(qsub_path.stat().st_mode | stat.S_IEXEC)
    qstat_path = Path("qstat")
    qstat_path.write_text(
        "#!/bin/sh"
        + dedent(
            f"""
            count=0
            if [ -f counter_file ]; then
                count=$(cat counter_file)
            fi
            echo "$((count+1))" > counter_file
            if [ $count -ge 3 ]; then
                json_flag_set=false;
                while [ "$#" -gt 0 ]; do
                    case "$1" in
                        -Fjson)
                            json_flag_set=true
                            ;;
                    esac
                    shift
                done
                if [ "$json_flag_set" = true ]; then
                    echo '{json.dumps({"Jobs": {"1": {"Job_Name": "1", "job_state": "E", "Exit_status": "0"}}})}'
                else
                    echo "{QSTAT_HEADER}"; printf "{QSTAT_HEADER_FORMAT}" 1 foo someuser 0 E normal
                fi
            else
                echo "{error_message_to_output}" >&2
                exit 2
            fi
        """  # noqa: E501
        ),
        encoding="utf-8",
    )
    qstat_path.chmod(qstat_path.stat().st_mode | stat.S_IEXEC)


@pytest.mark.parametrize(
    "text_to_ignore",
    [
        "pbs_iff: cannot connect to host\npbs_iff: all reserved ports in use",
        "qstat: Invalid credential",
    ],
)
async def test_that_openpbs_driver_ignores_qstat_flakiness(
    text_to_ignore: str, create_mock_flaky_qstat, caplog, capsys
):
    caplog.set_level(logging.DEBUG)
    create_mock_flaky_qstat(error_message_to_output=text_to_ignore)
    driver = OpenPBSDriver()
    driver._poll_period = 0.01
    await driver.submit(0, "sleep")

    with contextlib.suppress(TypeError):
        await asyncio.wait_for(poll(driver, expected={0}), timeout=10)

    assert int(Path("counter_file").read_text(encoding="utf-8")) >= 3, (
        "polling did not occur, test setup failed"
    )

    assert text_to_ignore not in capsys.readouterr().out
    assert text_to_ignore not in capsys.readouterr().err
    assert text_to_ignore not in caplog.text


@pytest.mark.parametrize(
    "job_return_code",
    [
        pytest.param(0, id="realization_finished_successfully"),
        pytest.param(2, id="realization_finished_with_error"),
    ],
)
async def test_that_kill_does_not_log_error_for_finished_realization(
    job_return_code, caplog, capsys, tmp_path, monkeypatch
):
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    qsub_path = bin_path / "qsub"
    qsub_path.write_text("#!/bin/sh\necho '1'")
    qsub_path.chmod(qsub_path.stat().st_mode | stat.S_IEXEC)
    driver = OpenPBSDriver()
    await driver.submit(0, "echo 1")

    async def mock_poll():
        job_id = driver._iens2jobid[0]
        driver._finished_job_ids.add(job_id)
        await driver._process_job_update(
            job_id, FinishedJob(job_state="F", returncode=job_return_code)
        )

    driver.poll = mock_poll
    await driver.poll()
    await driver.kill([0])

    assert "kill" not in capsys.readouterr().out
    assert "kill" not in capsys.readouterr().err
    assert "kill" not in caplog.text


def test_create_job_class_raises_error_on_invalid_state():
    with pytest.raises(TypeError, match=r"Invalid job state"):
        invalid_job_dict = {"job_state": "foobar"}
        _create_job_class(invalid_job_dict)


@pytest.mark.usefixtures("capturing_qsub")
async def test_submit_project_code():
    project_code = "testing+testing123"
    driver = OpenPBSDriver(project_code=project_code)
    await driver.submit(0, "sleep")
    assert f" -A {project_code} " in Path("captured_qsub_args").read_text(
        encoding="utf-8"
    )


@pytest.fixture(autouse=True)
def mock_openpbs(pytestconfig, monkeypatch, tmp_path):
    if pytestconfig.getoption("openpbs"):
        # User provided --openpbs, which means we should use the actual OpenPBS
        # cluster without mocking anything.
        return
    mock_bin(monkeypatch, tmp_path)


@pytest.fixture()
def queue_name_config():
    if queue_name := os.getenv("_ERT_TESTS_DEFAULT_QUEUE_NAME"):
        return f"\nQUEUE_OPTION TORQUE QUEUE {queue_name}"
    return ""


async def mock_failure(message, *args, **kwargs):
    raise RuntimeError(message)


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_poly_case")
def test_openpbs_driver_with_poly_example_failing_submit_fails_ert_and_propagates_exception_to_user(  # noqa: E501
    monkeypatch, caplog, queue_name_config
):
    monkeypatch.setattr(
        OpenPBSDriver, "submit", partial(mock_failure, "Submit job failed")
    )
    with open("poly.ert", mode="a+", encoding="utf-8") as f:
        f.write("QUEUE_SYSTEM TORQUE\nNUM_REALIZATIONS 2")
        f.write(queue_name_config)
    with pytest.raises(ErtCliError):
        run_cli(
            ENSEMBLE_EXPERIMENT_MODE,
            "poly.ert",
        )
    assert "RuntimeError: Submit job failed" in caplog.text


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_poly_case")
def test_openpbs_driver_with_poly_example_failing_poll_fails_ert_and_propagates_exception_to_user(  # noqa: E501
    monkeypatch, caplog, queue_name_config
):
    monkeypatch.setattr(EnsembleEvaluator, "BATCHING_INTERVAL", 0.05)
    monkeypatch.setattr(Scheduler, "BATCH_KILLING_INTERVAL", 0.01)
    monkeypatch.setattr(
        OpenPBSDriver, "poll", partial(mock_failure, "Status polling failed")
    )
    with open("poly.ert", mode="a+", encoding="utf-8") as f:
        f.write("QUEUE_SYSTEM TORQUE\nNUM_REALIZATIONS 2")
        f.write(queue_name_config)
    with pytest.raises(ErtCliError):
        run_cli(
            ENSEMBLE_EXPERIMENT_MODE,
            "poly.ert",
        )
    assert "RuntimeError: Status polling failed" in caplog.text
