import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

import pytest

from ert.scheduler.driver import SIGNAL_OFFSET, Driver
from ert.scheduler.local_driver import LocalDriver
from ert.scheduler.lsf_driver import LsfDriver
from ert.scheduler.openpbs_driver import OpenPBSDriver
from ert.scheduler.slurm_driver import SlurmDriver
from tests.ert.utils import poll

from .conftest import mock_bin


@pytest.fixture(params=[LocalDriver, LsfDriver, OpenPBSDriver, SlurmDriver])
def driver(request, pytestconfig, monkeypatch, tmp_path):
    class_ = request.param
    queue_name = None

    # It's not possible to dynamically choose a pytest fixture in a fixture, so
    # we copy some code here
    if class_ is OpenPBSDriver and pytestconfig.getoption("openpbs"):
        # User provided --openpbs, which means we should use the actual OpenPBS
        # cluster without mocking anything.
        if str(tmp_path).startswith("/tmp"):
            print(
                "Please use --basetemp option to pytest, PBS tests needs a shared disk"
            )
            sys.exit(1)
        queue_name = os.getenv("_ERT_TESTS_DEFAULT_QUEUE_NAME")
    elif class_ is LsfDriver and pytestconfig.getoption("lsf"):
        # User provided --lsf, which means we should use the actual LSF
        # cluster without mocking anything.""
        if str(tmp_path).startswith("/tmp"):
            print(
                "Please use --basetemp option to pytest, "
                "the real LSF cluster needs a shared disk"
            )
            sys.exit(1)
    elif class_ is SlurmDriver and pytestconfig.getoption("slurm"):
        pass
    else:
        mock_bin(monkeypatch, tmp_path)

    if class_ is LocalDriver:
        return class_()
    return class_(queue_name=queue_name)


@pytest.mark.integration_test
async def test_submit(driver: Driver, tmp_path, job_name):
    os.chdir(tmp_path)
    await driver.submit(0, "sh", "-c", f"echo test > {tmp_path}/test", name=job_name)
    await poll(driver, {0})

    assert (tmp_path / "test").read_text(encoding="utf-8") == "test\n"


@pytest.mark.integration_test
async def test_submit_something_that_fails(driver: Driver, tmp_path, job_name):
    os.chdir(tmp_path)
    finished_called = False

    expected_returncode = 42

    async def finished(iens, returncode):
        assert iens == 0
        assert returncode == expected_returncode

        nonlocal finished_called
        finished_called = True

    await driver.submit(
        0,
        "sh",
        "-c",
        f"exit {expected_returncode}",
        runpath=tmp_path,
        name=job_name,
    )
    await poll(driver, {0}, finished=finished)

    assert finished_called


@pytest.mark.integration_test
async def test_kill_gives_correct_state(driver: Driver, tmp_path, request):
    os.chdir(tmp_path)
    aborted_called = False

    if isinstance(driver, SlurmDriver):
        expected_returncodes = [
            0,  # real Slurm
            271,  # mocked Slurm
        ]
    else:
        expected_returncodes = [
            SIGNAL_OFFSET + signal.SIGTERM,
            SIGNAL_OFFSET + signal.SIGINT,
            256 + signal.SIGKILL,
            256 + signal.SIGTERM,
        ]

    async def started(iens):
        nonlocal driver
        await driver.kill(iens)

    async def finished(iens, returncode):
        assert iens == 0
        assert returncode in expected_returncodes

        nonlocal aborted_called
        aborted_called = True

    job_name: str = request.node.name.replace("[", "__").replace("]", "__")
    await driver.submit(0, "sh", "-c", "sleep 60; exit 2", name=job_name)
    await poll(driver, {0}, started=started, finished=finished)
    assert aborted_called


@pytest.mark.integration_test
@pytest.mark.flaky(reruns=10)
async def test_repeated_submit_same_iens(driver: Driver, tmp_path):
    """Submits are allowed to be repeated for the same iens, and are to be
    handled according to FIFO, but this order cannot be guaranteed as it depends
    on the host operating system."""
    os.chdir(tmp_path)
    await driver.submit(
        0,
        "sh",
        "-c",
        f"echo submit1 > {tmp_path}/submissionrace; touch {tmp_path}/submit1",
        name="submit1",
    )
    await driver.submit(
        0,
        "sh",
        "-c",
        f"echo submit2 > {tmp_path}/submissionrace; touch {tmp_path}/submit2",
        name="submit2",
    )
    # Wait until both submissions have done their thing:
    while not Path("submit1").exists() or not Path("submit2").exists():  # noqa: ASYNC110
        await asyncio.sleep(0.1)
    assert Path("submissionrace").read_text(encoding="utf-8") == "submit2\n"


@pytest.mark.integration_test
@pytest.mark.flaky(reruns=5)
async def test_kill_actually_kills(driver: Driver, tmp_path, pytestconfig):
    os.chdir(tmp_path)
    if (
        (isinstance(driver, LsfDriver) and pytestconfig.getoption("lsf"))  # noqa: PLR0916
        or (isinstance(driver, OpenPBSDriver) and pytestconfig.getoption("openpbs"))
        or (isinstance(driver, SlurmDriver) and pytestconfig.getoption("slurm"))
    ):
        # Allow more time when tested on a real compute cluster to avoid false positives.
        job_kill_window = 60
        test_grace_time = 120
    else:
        job_kill_window = 5  # Busy test nodes require a long kill window
        test_grace_time = 8

    async def kill_job_once_started(iens):
        nonlocal driver
        await driver.kill(iens)

    await driver.submit(
        0,
        "sh",
        "-c",
        f"sleep {job_kill_window}; touch {tmp_path}/survived",
        name="kill_me",
    )
    await poll(driver, {0}, started=kill_job_once_started)

    # Give the script a chance to finish if it is running
    await asyncio.sleep(test_grace_time)
    assert not Path("survived").exists(), "Job should have been killed"


@pytest.mark.integration_test
async def test_num_cpu_sets_env_variables(driver: Driver, tmp_path, job_name):
    """The intention of this check is to verify that the driver sets up
    the num_cpu requirement correctly for the relevant queue system.

    How this can be verified depends on the queue system, there is no single
    environment variable that they all set."""
    if isinstance(driver, LocalDriver):
        pytest.skip("LocalDriver has no NUM_CPU concept")
    os.chdir(tmp_path)
    await driver.submit(
        0,
        "sh",
        "-c",
        f"env | grep -e PROCESS -e CPU -e THREAD > {tmp_path}/env",
        name=job_name,
        num_cpu=2,
    )
    await poll(driver, {0})

    env_lines = Path(f"{tmp_path}/env").read_text(encoding="utf-8").splitlines()
    if isinstance(driver, SlurmDriver):
        assert "SLURM_JOB_CPUS_PER_NODE=2" in env_lines
        assert "SLURM_CPUS_ON_NODE=2" in env_lines
    elif isinstance(driver, LsfDriver):
        assert "LSB_MAX_NUM_PROCESSORS=2" in env_lines
    elif isinstance(driver, OpenPBSDriver):
        assert "OMP_NUM_THREADS=2" in env_lines
        assert "NCPUS=2" in env_lines


async def test_execute_with_retry_exits_on_filenotfounderror(driver: Driver, caplog):
    caplog.set_level(logging.DEBUG)
    invalid_cmd = ["/usr/bin/foo", "bar"]
    (succeeded, message) = await driver._execute_with_retry(
        invalid_cmd, total_attempts=3
    )

    # We log a retry message every time we retry
    assert "retry" not in str(caplog.text)
    assert not succeeded
    assert "No such file or directory" in message
    assert "/usr/bin/foo" in message


@pytest.mark.integration_test
async def test_poll_exits_on_filenotfounderror(driver: Driver, caplog):
    if isinstance(driver, LocalDriver):
        pytest.skip("LocalDriver does not poll")
    caplog.set_level(logging.DEBUG)
    invalid_cmd = ["/usr/bin/foo", "bar"]
    driver._bjobs_cmd = invalid_cmd
    driver._qstat_cmd = invalid_cmd
    driver._squeue = invalid_cmd
    driver._jobs = {"foo": "bar"}
    driver._non_finished_job_ids = ["foo"]
    await driver.poll()

    # We log a retry message every time we retry
    assert "retry" not in str(caplog.text)
    assert "No such file or directory" in str(caplog.text)
    assert "/usr/bin/foo" in str(caplog.text)
