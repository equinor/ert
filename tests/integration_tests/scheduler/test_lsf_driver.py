import asyncio
import json
import os
import stat
from pathlib import Path

import pytest

from ert.scheduler import LsfDriver
from ert.scheduler.lsf_driver import LSF_FAILED_JOB
from tests.utils import poll

from .conftest import mock_bin


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
    os.chdir(tmp_path)
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bjobs_path = bin_path / "bjobs"
    bjobs_path.write_text(
        "#!/bin/sh\n" 'echo "Job <$1> is not found"',
        encoding="utf-8",
    )
    bjobs_path.chmod(bjobs_path.stat().st_mode | stat.S_IEXEC)


@pytest.mark.parametrize("explicit_runpath", [(True), (False)])
async def test_lsf_info_file_in_runpath(explicit_runpath, tmp_path):
    os.chdir(tmp_path)
    driver = LsfDriver()
    (tmp_path / "some_runpath").mkdir()
    os.chdir(tmp_path)
    effective_runpath = tmp_path / "some_runpath" if explicit_runpath else tmp_path
    await driver.submit(
        0,
        "sh",
        "-c",
        "exit 0",
        runpath=tmp_path / "some_runpath" if explicit_runpath else None,
    )

    await poll(driver, {0})

    effective_runpath = tmp_path / "some_runpath" if explicit_runpath else tmp_path
    assert json.loads(
        (effective_runpath / "lsf_info.json").read_text(encoding="utf-8")
    ).keys() == {"job_id"}


async def test_job_name(tmp_path):
    os.chdir(tmp_path)
    driver = LsfDriver()
    iens: int = 0
    await driver.submit(iens, "sh", "-c", "sleep 99", name="my_job")
    jobid = driver._iens2jobid[iens]
    bjobs_process = await asyncio.create_subprocess_exec(
        "bjobs",
        jobid,
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await bjobs_process.communicate()
    assert "my_job" in stdout.decode()


@pytest.mark.integration_test
async def test_submit_to_named_queue(tmp_path, caplog):
    """If the environment variable _ERT_TEST_ALTERNATIVE_QUEUE is defined
    a job will be attempted submitted to that queue.

    As Ert does not keep track of which queue a job is executed in, we can only
    test for success for the job."""
    os.chdir(tmp_path)
    driver = LsfDriver(queue_name=os.getenv("_ERT_TESTS_ALTERNATIVE_QUEUE"))
    await driver.submit(0, "sh", "-c", f"echo test > {tmp_path}/test")
    await poll(driver, {0})

    assert (tmp_path / "test").read_text(encoding="utf-8") == "test\n"


@pytest.mark.parametrize(
    "actual_returncode, returncode_that_ert_sees",
    [
        ([0, 0]),
        ([1, LSF_FAILED_JOB]),
        ([2, LSF_FAILED_JOB]),
        ([255, LSF_FAILED_JOB]),
        ([256, 0]),  # return codes are 8 bit.
    ],
)
async def test_lsf_driver_masks_returncode(
    actual_returncode, returncode_that_ert_sees, tmp_path
):
    """actual_returncode is the returncode from job_dispatch.py (or whatever is submitted)

    The LSF driver is not picking up this returncode, it will only look at the
    status the job obtains through bjobs, which is success/failure.
    """
    os.chdir(tmp_path)
    driver = LsfDriver()

    async def finished(iens, returncode):
        assert iens == 0
        assert returncode == returncode_that_ert_sees

    await driver.submit(0, "sh", "-c", f"exit {actual_returncode}")
    await poll(driver, {0}, finished=finished)


async def test_submit_with_resource_requirement(tmp_path):
    resource_requirement = "select[cs && x86_64Linux]"
    driver = LsfDriver(resource_requirement=resource_requirement)
    await driver.submit(0, "sh", "-c", f"echo test>{tmp_path}/test")
    await poll(driver, {0})

    assert (tmp_path / "test").read_text(encoding="utf-8") == "test\n"


async def test_polling_bhist_fallback(not_found_bjobs):
    driver = LsfDriver()
    Path("mock_jobs").mkdir()
    Path("mock_jobs/pendingtimemillis").write_text("100")
    driver._poll_period = 0.01
    await driver.submit(0, "sh", "-c", "sleep 1")
    await poll(driver, {0})
