import os
import stat
from pathlib import Path
from threading import BoundedSemaphore
from typing import List, TypedDict
from unittest.mock import MagicMock

import pytest

from ert.config import QueueSystem
from ert.job_queue import Driver, JobQueue, JobQueueManager, JobQueueNode, JobStatus


class Config(TypedDict):
    num_cpu: int
    job_name: str
    run_path: str


DUMMY_CONFIG: Config = {
    "num_cpu": 1,
    "job_name": "dummy_job_{}",
    "run_path": "dummy_path_{}",
}


MOCK_BSUB = """#!/bin/sh
echo "$@" > test.out
"""
"""A dummy bsub script that instead of submitting a job to an LSF cluster
writes the arguments it got to a file called test.out, mimicking what
an actual cluster node might have done."""


def create_local_queue(
    executable_script: str, max_submit: int = 2, num_realizations: int = 10
):
    driver = Driver(driver_type=QueueSystem.LOCAL)
    job_queue = JobQueue(driver, max_submit=max_submit)

    for iens in range(num_realizations):
        Path(DUMMY_CONFIG["run_path"].format(iens)).mkdir()
        job = JobQueueNode(
            job_script=executable_script,
            job_name=DUMMY_CONFIG["job_name"].format(iens),
            run_path=os.path.realpath(DUMMY_CONFIG["run_path"].format(iens)),
            num_cpu=DUMMY_CONFIG["num_cpu"],
            status_file=job_queue.status_file,
            exit_file=job_queue.exit_file,
            run_arg=MagicMock(),
        )
        job_queue.add_job(job, iens)
    return job_queue


@pytest.mark.usefixtures("use_tmpdir", "mock_fm_ok")
def test_num_cpu_submitted_correctly_lsf(tmpdir, simple_script):
    """Assert that num_cpu from the ERT configuration is passed on to the bsub
    command used to submit jobs to LSF"""
    os.putenv("PATH", os.getcwd() + ":" + os.getenv("PATH"))
    driver = Driver(driver_type=QueueSystem.LSF)

    bsub = Path("bsub")
    bsub.write_text(MOCK_BSUB, encoding="utf-8")
    bsub.chmod(stat.S_IRWXU)

    job_id = 0
    num_cpus = 4
    os.mkdir(DUMMY_CONFIG["run_path"].format(job_id))

    job = JobQueueNode(
        job_script=simple_script,
        job_name=DUMMY_CONFIG["job_name"].format(job_id),
        run_path=os.path.realpath(DUMMY_CONFIG["run_path"].format(job_id)),
        num_cpu=4,
        status_file="STATUS",
        exit_file="ERROR",
        run_arg=MagicMock(),
    )

    pool_sema = BoundedSemaphore(value=2)
    job.run(driver, pool_sema)
    job.stop()
    job.wait_for()

    bsub_argv: List[str] = Path("test.out").read_text(encoding="utf-8").split()

    found_cpu_arg = False
    for arg_i, arg in enumerate(bsub_argv):
        if arg == "-n":
            # Check that the driver submitted the correct number
            # of cpus:
            assert bsub_argv[arg_i + 1] == str(num_cpus)
            found_cpu_arg = True

    assert found_cpu_arg is True


def test_execute_queue(tmpdir, monkeypatch, mock_fm_ok, simple_script):
    monkeypatch.chdir(tmpdir)
    job_queue = create_local_queue(simple_script)
    manager = JobQueueManager(job_queue)
    manager.execute_queue()

    assert len(mock_fm_ok.mock_calls) == len(job_queue.job_list)


@pytest.mark.parametrize("max_submit_num", [1, 2, 3])
def test_max_submit_reached(tmpdir, max_submit_num, monkeypatch, failing_script):
    """Check that the JobQueueManager will submit exactly the maximum number of
    resubmissions in the case of scripts that fail."""
    monkeypatch.chdir(tmpdir)
    num_realizations = 2
    job_queue = create_local_queue(
        failing_script,
        max_submit=max_submit_num,
        num_realizations=num_realizations,
    )

    manager = JobQueueManager(job_queue)
    manager.execute_queue()

    assert (
        Path("one_byte_pr_invocation").stat().st_size
        == max_submit_num * num_realizations
    )

    assert manager.isRunning() is False

    for job in job_queue.job_list:
        # one for every realization
        assert job.status == JobStatus.FAILED
        assert job.submit_attempt == job_queue.max_submit


@pytest.mark.parametrize("max_submit_num", [1, 2, 3])
def test_kill_queue(tmpdir, max_submit_num, monkeypatch, simple_script):
    monkeypatch.chdir(tmpdir)
    job_queue = create_local_queue(simple_script, max_submit=max_submit_num)
    manager = JobQueueManager(job_queue)
    job_queue.kill_all_jobs()
    manager.execute_queue()

    assert not Path("STATUS").exists()
    for job in job_queue.job_list:
        assert job.status == JobStatus.FAILED
