import asyncio
import json
import logging
import os
import random
import re
import stat
import string
import sys
from pathlib import Path

import pytest

from ert.scheduler import LsfDriver
from ert.scheduler.driver import SIGNAL_OFFSET
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


async def test_lsf_stdout_file(tmp_path, job_name):
    os.chdir(tmp_path)
    driver = LsfDriver()
    await driver.submit(0, "sh", "-c", "echo yay", name=job_name)
    await poll(driver, {0})
    lsf_stdout = Path(f"{job_name}.LSF-stdout").read_text(encoding="utf-8")
    assert Path(
        f"{job_name}.LSF-stdout"
    ).exists(), "LSF system did not write output file"

    assert "Sender: " in lsf_stdout, "LSF stdout should always start with 'Sender:'"
    assert "The output (if any) follows:" in lsf_stdout
    assert "yay" in lsf_stdout


async def test_lsf_dumps_stderr_to_file(tmp_path, job_name):
    os.chdir(tmp_path)
    driver = LsfDriver()
    failure_message = "failURE"
    await driver.submit(0, "sh", "-c", f"echo {failure_message} >&2", name=job_name)
    await poll(driver, {0})
    assert Path(
        f"{job_name}.LSF-stderr"
    ).exists(), "LSF system did not write stderr file"

    assert (
        Path(f"{job_name}.LSF-stderr").read_text(encoding="utf-8").strip()
        == failure_message
    )


def generate_random_text(size):
    letters = string.ascii_letters
    return "".join(random.choice(letters) for i in range(size))


@pytest.mark.parametrize("tail_chars_to_read", [(5), (50), (500), (700)])
async def test_lsf_can_retrieve_stdout_and_stderr(
    tmp_path, job_name, tail_chars_to_read
):
    os.chdir(tmp_path)
    driver = LsfDriver()
    num_written_characters = 600
    _out = generate_random_text(num_written_characters)
    _err = generate_random_text(num_written_characters)
    await driver.submit(0, "sh", "-c", f"echo {_out} && echo {_err} >&2", name=job_name)
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


async def test_lsf_cannot_retrieve_stdout_and_stderr(tmp_path, job_name):
    os.chdir(tmp_path)
    driver = LsfDriver()
    num_written_characters = 600
    _out = generate_random_text(num_written_characters)
    _err = generate_random_text(num_written_characters)
    await driver.submit(0, "sh", "-c", f"echo {_out} && echo {_err} >&2", name=job_name)
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


@pytest.mark.parametrize("explicit_runpath", [(True), (False)])
async def test_lsf_info_file_in_runpath(explicit_runpath, tmp_path, job_name):
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
        name=job_name,
    )

    await poll(driver, {0})

    effective_runpath = tmp_path / "some_runpath" if explicit_runpath else tmp_path
    assert json.loads(
        (effective_runpath / "lsf_info.json").read_text(encoding="utf-8")
    ).keys() == {"job_id"}


@pytest.mark.integration_test
async def test_submit_to_named_queue(tmp_path, caplog, job_name):
    """If the environment variable _ERT_TEST_ALTERNATIVE_QUEUE is defined
    a job will be attempted submitted to that queue.

    As Ert does not keep track of which queue a job is executed in, we can only
    test for success for the job."""
    os.chdir(tmp_path)
    driver = LsfDriver(queue_name=os.getenv("_ERT_TESTS_ALTERNATIVE_QUEUE"))
    await driver.submit(0, "sh", "-c", f"echo test > {tmp_path}/test", name=job_name)
    await poll(driver, {0})

    assert (tmp_path / "test").read_text(encoding="utf-8") == "test\n"


@pytest.mark.usefixtures("use_tmpdir")
async def test_submit_with_resource_requirement(job_name):
    resource_requirement = "select[cs && x86_64Linux]"
    driver = LsfDriver(resource_requirement=resource_requirement)
    await driver.submit(0, "sh", "-c", "echo test>test", name=job_name)
    await poll(driver, {0})

    assert Path("test").read_text(encoding="utf-8") == "test\n"


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
    assert matches and matches[1] == str(
        num_cpu
    ), f"Could not verify processor allocation from stdout: {stdout}, stderr: {stderr}"

    assert Path("test").read_text(encoding="utf-8") == "test\n"


async def test_polling_bhist_fallback(not_found_bjobs, caplog, job_name):
    caplog.set_level(logging.DEBUG)
    driver = LsfDriver()
    Path("mock_jobs").mkdir()
    Path("mock_jobs/pendingtimemillis").write_text("100")
    driver._poll_period = 0.01

    bhist_called = False
    original_bhist_method = driver._poll_once_by_bhist

    def mock_poll_once_by_bhist(*args, **kwargs):
        nonlocal bhist_called
        bhist_called = True
        return original_bhist_method(*args, **kwargs)

    driver._poll_once_by_bhist = mock_poll_once_by_bhist

    await driver.submit(0, "sh", "-c", "sleep 1", name=job_name)
    job_id = list(driver._iens2jobid.values())[0]
    await poll(driver, {0})
    assert "bhist is used" in caplog.text
    assert bhist_called
    assert job_id in driver._bhist_cache


async def test_kill_before_submit_is_finished(
    tmp_path, monkeypatch, caplog, pytestconfig
):
    os.chdir(tmp_path)

    if pytestconfig.getoption("lsf"):
        # Allow more time when tested on a real compute cluster to avoid false positives.
        job_kill_window = 10
        test_grace_time = 20
    elif sys.platform.startswith("darwin"):
        # Mitigate flakiness on low-power test nodes
        job_kill_window = 5
        test_grace_time = 10
    else:
        job_kill_window = 1
        test_grace_time = 2

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

    # Allow submit and kill to be interleaved by asyncio by issuing
    # submit() in its own asyncio Task:
    asyncio.create_task(
        driver.submit(
            # The sleep is the time window in which we can kill the job before
            # the unwanted finish message appears on disk.
            0,
            "sh",
            "-c",
            f"sleep {job_kill_window}; touch {tmp_path}/survived",
        )
    )
    await asyncio.sleep(0.01)  # Allow submit task to start executing
    await driver.kill(0)  # This will wait until the submit is done and then kill

    async def finished(iens: int, returncode: int):
        SIGTERM = 15
        assert iens == 0
        assert returncode == SIGNAL_OFFSET + SIGTERM

    await poll(driver, {0}, finished=finished)

    assert "ERROR" not in str(caplog.text)
    await asyncio.sleep(test_grace_time)
    assert not Path("survived").exists(), "Job should have been killed"
