import asyncio
import logging
import os
import random
import stat
import string
import sys
from pathlib import Path

import pytest

from ert.scheduler import SlurmDriver
from tests.utils import poll

from .conftest import mock_bin


@pytest.fixture(autouse=True)
def mock_slurm(pytestconfig, monkeypatch, tmp_path):
    if pytestconfig.getoption("slurm"):
        # User provided --slurm, which means we should use an actual Slurm
        # cluster without mocking anything.""
        return
    mock_bin(monkeypatch, tmp_path)


async def test_slurm_stdout_file(tmp_path, job_name):
    os.chdir(tmp_path)
    driver = SlurmDriver()
    await driver.submit(0, "sh", "-c", "echo yay", name=job_name)
    await poll(driver, {0})
    slurm_stdout = Path(f"{job_name}.stdout").read_text(encoding="utf-8")
    assert Path(f"{job_name}.stdout").exists(), "Slurm system did not write output file"
    assert "yay" in slurm_stdout


async def test_slurm_dumps_stderr_to_file(tmp_path, job_name):
    os.chdir(tmp_path)
    driver = SlurmDriver()
    failure_message = "failURE"
    await driver.submit(0, "sh", "-c", f"echo {failure_message} >&2", name=job_name)
    await poll(driver, {0})
    assert Path(f"{job_name}.stderr").exists(), "Slurm system did not write stderr file"

    assert (
        Path(f"{job_name}.stderr").read_text(encoding="utf-8").strip()
        == failure_message
    )


def generate_random_text(size):
    letters = string.ascii_letters
    return "".join(random.choice(letters) for _ in range(size))


@pytest.mark.parametrize("tail_chars_to_read", [(5), (50), (500), (700)])
async def test_slurm_can_retrieve_stdout_and_stderr(
    tmp_path, job_name, tail_chars_to_read
):
    os.chdir(tmp_path)
    driver = SlurmDriver()
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

    stderr_txt = Path(f"{job_name}.stderr").read_text(encoding="utf-8").strip()
    stdout_txt = Path(f"{job_name}.stdout").read_text(encoding="utf-8").strip()

    assert stderr_txt[-min(tail_chars_to_read, num_written_characters) + 2 :] in message
    assert stdout_txt[-min(tail_chars_to_read, num_written_characters) + 2 :] in message


@pytest.mark.integration_test
async def test_submit_to_named_queue(tmp_path, job_name):
    """If the environment variable _ERT_TEST_ALTERNATIVE_QUEUE is defined
    a job will be attempted submitted to that queue.

    * Note that what is called a "queue" in Ert is a "partition" in Slurm lingo.

    As Ert does not keep track of which queue a job is executed in, we can only
    test for success for the job."""
    os.chdir(tmp_path)
    driver = SlurmDriver(queue_name=os.getenv("_ERT_TESTS_ALTERNATIVE_QUEUE"))
    await driver.submit(0, "sh", "-c", f"echo test > {tmp_path}/test", name=job_name)
    await poll(driver, {0})

    assert (tmp_path / "test").read_text(encoding="utf-8") == "test\n"


@pytest.mark.usefixtures("use_tmpdir")
async def test_submit_with_num_cpu(pytestconfig, job_name):
    if not pytestconfig.getoption("slurm"):
        return

    num_cpu = 2
    driver = SlurmDriver()
    await driver.submit(0, "sh", "-c", "echo test>test", name=job_name, num_cpu=num_cpu)
    job_id = driver._iens2jobid[0]
    await poll(driver, {0})

    process = await asyncio.create_subprocess_exec(
        "scontrol",
        "show",
        "job",
        job_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    assert " NumCPUs=2 " in stdout.decode(
        errors="ignore"
    ), f"Could not verify processor allocation from stdout: {stdout}, stderr: {stderr}"

    assert Path("test").read_text(encoding="utf-8") == "test\n"


@pytest.mark.flaky(reruns=3)
async def test_kill_before_submit_is_finished(
    tmp_path, monkeypatch, caplog, pytestconfig
):
    os.chdir(tmp_path)

    if pytestconfig.getoption("slurm"):
        # Allow more time when tested on a real compute cluster to avoid false positives.
        job_kill_window = 5
        test_grace_time = 10
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
    sbatch_path = bin_path / "slow_sbatch"
    sbatch_path.write_text(
        "#!/bin/sh\nsleep 0.1\nsbatch $@",
        encoding="utf-8",
    )
    sbatch_path.chmod(sbatch_path.stat().st_mode | stat.S_IEXEC)

    caplog.set_level(logging.DEBUG)
    driver = SlurmDriver(sbatch_cmd="slow_sbatch")

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
        assert iens == 0
        # Slurm assigns returncode 0 even when they are killed.
        assert returncode == 0

    await poll(driver, {0}, finished=finished)

    # In case the return value of the killed job is correct but the submitted
    # shell script is still running for whatever reason, a file called
    # "survived" will appear on disk. Wait for it, and then ensure it is not
    # there.
    assert test_grace_time > job_kill_window, "Wrong test setup"
    await asyncio.sleep(test_grace_time)
    assert not Path(
        "survived"
    ).exists(), "The process children of the job should also have been killed"
