import asyncio
import json
import os
from pathlib import Path

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


@pytest.mark.parametrize("runpath_supplied", [(True), (False)])
async def test_lsf_info_file_in_runpath(runpath_supplied, tmp_path):
    driver = LsfDriver()
    os.chdir(tmp_path)
    if runpath_supplied:
        await driver.submit(0, "exit 0", runpath=str(tmp_path))
    else:
        await driver.submit(0, "exit 0")

    await poll(driver, {0})

    if runpath_supplied:
        assert json.loads(
            (tmp_path / "lsf_info.json").read_text(encoding="utf-8")
        ).keys() == {"job_id"}

    else:
        assert not Path("lsf_info.json").exists()


async def test_job_name():
    driver = LsfDriver()
    iens: int = 0
    await driver.submit(iens, "sleep 99", name="my_job_name")
    jobid = driver._iens2jobid[iens]
    bjobs_process = await asyncio.create_subprocess_exec(
        "bjobs",
        jobid,
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await bjobs_process.communicate()
    assert "my_job_name" in stdout.decode()


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

    await driver.submit(0, f"exit {actual_returncode}")
    await poll(driver, {0}, finished=finished)
