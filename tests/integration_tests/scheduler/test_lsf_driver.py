import asyncio
import json
import os

import pytest

from ert.scheduler import LsfDriver
from tests.utils import poll

from .conftest import mock_bin


@pytest.fixture(autouse=True)
def mock_lsf(pytestconfig, monkeypatch, tmp_path):
    if pytestconfig.getoption("lsf"):
        # User provided --lsf, which means we should use the actual LSF
        # cluster without mocking anything.""
        return
    mock_bin(monkeypatch, tmp_path)


@pytest.mark.parametrize("explicit_runpath", [(True), (False)])
async def test_lsf_info_file_in_runpath(explicit_runpath, tmp_path):
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


async def test_job_name():
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
        ([1, 1]),
        ([2, 1]),
        ([255, 1]),
        ([256, 0]),  # return codes are 8 bit.
    ],
)
async def test_lsf_driver_masks_returncode(actual_returncode, returncode_that_ert_sees):
    """actual_returncode is the returncode from job_dispatch.py (or whatever is submitted)

    The LSF driver is not picking up this returncode, it will only look at the
    status the job obtains through bjobs, which is success/failure.
    """
    driver = LsfDriver()

    async def finished(iens, returncode, aborted):
        assert iens == 0
        assert returncode == returncode_that_ert_sees
        assert aborted == (returncode_that_ert_sees != 0)

    await driver.submit(0, "sh", "-c", f"exit {actual_returncode}")
    await poll(driver, {0}, finished=finished)
