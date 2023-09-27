import json
import stat
import time
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Callable, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from ert.config import QueueSystem
from ert.job_queue import Driver, JobQueue, JobQueueNode, JobStatus
from ert.run_arg import RunArg
from ert.storage import EnsembleAccessor


def wait_for(
    func: Callable, target: Any = True, interval: float = 0.1, timeout: float = 30
):
    """Sleeps (with timeout) until the provided function returns the provided target"""
    t = 0.0
    while func() != target:
        time.sleep(interval)
        t += interval
        if t >= timeout:
            raise AssertionError(
                "Timeout reached in wait_for "
                f"(function {func.__name__}, timeout {timeout}) "
            )


DUMMY_CONFIG: Dict[str, Any] = {
    "job_script": "job_script.py",
    "num_cpu": 1,
    "job_name": "dummy_job_{}",
    "run_path": "dummy_path_{}",
}


@pytest.fixture
def never_ending_script(tmp_path):
    NEVER_ENDING_SCRIPT = """#!/usr/bin/env python
import time
while True:
    time.sleep(0.5)
    """
    fout = Path(tmp_path / "never_ending_job_script")
    fout.write_text(NEVER_ENDING_SCRIPT, encoding="utf-8")
    fout.chmod(stat.S_IRWXU | stat.S_IRWXO | stat.S_IRWXG)
    yield str(fout)


def create_local_queue(
    executable_script: str,
    max_submit: int = 1,
    max_runtime: Optional[int] = None,
    callback_timeout: Optional["Callable[[int], None]"] = None,
):
    driver = Driver(driver_type=QueueSystem.LOCAL)
    job_queue = JobQueue(driver, max_submit=max_submit)

    for iens in range(10):
        Path(DUMMY_CONFIG["run_path"].format(iens)).mkdir(exist_ok=False)
        job = JobQueueNode(
            job_script=executable_script,
            num_cpu=DUMMY_CONFIG["num_cpu"],
            status_file=job_queue.status_file,
            exit_file=job_queue.exit_file,
            run_arg=RunArg(
                str(iens),
                MagicMock(spec=EnsembleAccessor),
                0,
                0,
                DUMMY_CONFIG["run_path"].format(iens),
                DUMMY_CONFIG["job_name"].format(iens),
            ),
            max_runtime=max_runtime,
            callback_timeout=callback_timeout,
        )

        job_queue.add_job(job, iens)

    return job_queue


def start_all(job_queue, sema_pool):
    job = job_queue.fetch_next_waiting()
    while job is not None:
        job.run(job_queue.driver, sema_pool, job_queue.max_submit)
        job = job_queue.fetch_next_waiting()


def test_kill_jobs(tmpdir, monkeypatch, never_ending_script):
    monkeypatch.chdir(tmpdir)
    job_queue = create_local_queue(never_ending_script)

    assert job_queue.queue_size == 10
    assert job_queue.is_active()

    pool_sema = BoundedSemaphore(value=10)
    start_all(job_queue, pool_sema)

    # Make sure NEVER_ENDING_SCRIPT has started:
    wait_for(job_queue.is_active)

    # Ask the job to stop:
    for job in job_queue.job_list:
        job.stop()

    wait_for(job_queue.is_active, target=False)

    # pylint: disable=protected-access
    job_queue._differ.transition(job_queue.job_list)

    for q_index, job in enumerate(job_queue.job_list):
        assert job.queue_status == JobStatus.IS_KILLED
        iens = job_queue._differ.qindex_to_iens(q_index)
        assert job_queue.snapshot()[iens] == str(JobStatus.IS_KILLED)

    for job in job_queue.job_list:
        job.wait_for()


def test_add_jobs(tmpdir, monkeypatch, simple_script):
    monkeypatch.chdir(tmpdir)
    job_queue = create_local_queue(simple_script)

    assert job_queue.queue_size == 10
    assert job_queue.is_active()
    assert job_queue.fetch_next_waiting() is not None

    pool_sema = BoundedSemaphore(value=10)
    start_all(job_queue, pool_sema)

    for job in job_queue.job_list:
        job.stop()

    wait_for(job_queue.is_active, target=False)

    for job in job_queue.job_list:
        job.wait_for()


def test_failing_jobs(tmpdir, monkeypatch, failing_script):
    monkeypatch.chdir(tmpdir)
    job_queue = create_local_queue(failing_script, max_submit=1)

    assert job_queue.queue_size == 10
    assert job_queue.is_active()

    pool_sema = BoundedSemaphore(value=10)
    start_all(job_queue, pool_sema)

    wait_for(job_queue.is_active, target=False)

    for job in job_queue.job_list:
        job.wait_for()

    # pylint: disable=protected-access
    job_queue._differ.transition(job_queue.job_list)

    assert job_queue.fetch_next_waiting() is None

    for q_index, job in enumerate(job_queue.job_list):
        assert job.queue_status == JobStatus.FAILED
        iens = job_queue._differ.qindex_to_iens(q_index)
        assert job_queue.snapshot()[iens] == str(JobStatus.FAILED)


def test_timeout_jobs(tmpdir, monkeypatch, never_ending_script):
    monkeypatch.chdir(tmpdir)
    job_numbers = set()

    mock_callback = MagicMock()

    job_queue = create_local_queue(
        never_ending_script,
        max_submit=1,
        max_runtime=5,
        callback_timeout=mock_callback,
    )

    assert job_queue.queue_size == 10
    assert job_queue.is_active()

    pool_sema = BoundedSemaphore(value=10)
    start_all(job_queue, pool_sema)

    # Make sure NEVER_ENDING_SCRIPT jobs have started:
    wait_for(job_queue.is_active)

    # Wait for the timeout to kill them:
    wait_for(job_queue.is_active, target=False)

    # pylint: disable=protected-access
    job_queue._differ.transition(job_queue.job_list)

    for q_index, job in enumerate(job_queue.job_list):
        assert job.queue_status == JobStatus.IS_KILLED
        iens = job_queue._differ.qindex_to_iens(q_index)
        assert job_queue.snapshot()[iens] == str(JobStatus.IS_KILLED)

    assert len(mock_callback.mock_calls) == 20

    for job in job_queue.job_list:
        job.wait_for()


def test_add_dispatch_info(tmpdir, monkeypatch, simple_script):
    monkeypatch.chdir(tmpdir)
    job_queue = create_local_queue(simple_script)
    ens_id = "some_id"
    cert = "My very nice cert"
    token = "my_super_secret_token"
    dispatch_url = "wss://example.org"
    cert_file = ".ee.pem"
    runpaths = [Path(DUMMY_CONFIG["run_path"].format(iens)) for iens in range(10)]
    for runpath in runpaths:
        (runpath / "jobs.json").write_text(json.dumps({}), encoding="utf-8")
    job_queue.add_dispatch_information_to_jobs_file(
        ens_id=ens_id,
        dispatch_url=dispatch_url,
        cert=cert,
        token=token,
        experiment_id="experiment_id",
    )

    for runpath in runpaths:
        job_file_path = runpath / "jobs.json"
        content: dict = json.loads(job_file_path.read_text(encoding="utf-8"))
        assert content["step_id"] == 0
        assert content["dispatch_url"] == dispatch_url
        assert content["ee_token"] == token
        assert content["experiment_id"] == "experiment_id"

        assert content["ee_cert_path"] == str(runpath / cert_file)
        assert (runpath / cert_file).read_text(encoding="utf-8") == cert


def test_add_dispatch_info_cert_none(tmpdir, monkeypatch, simple_script):
    monkeypatch.chdir(tmpdir)
    job_queue = create_local_queue(simple_script)
    ens_id = "some_id"
    dispatch_url = "wss://example.org"
    cert = None
    token = None
    cert_file = ".ee.pem"
    runpaths = [Path(DUMMY_CONFIG["run_path"].format(iens)) for iens in range(10)]
    for runpath in runpaths:
        (runpath / "jobs.json").write_text(json.dumps({}), encoding="utf-8")
    job_queue.add_dispatch_information_to_jobs_file(
        ens_id=ens_id,
        dispatch_url=dispatch_url,
        cert=cert,
        token=token,
    )

    for runpath in runpaths:
        job_file_path = runpath / "jobs.json"
        content: dict = json.loads(job_file_path.read_text(encoding="utf-8"))
        assert content["step_id"] == 0
        assert content["dispatch_url"] == dispatch_url
        assert content["ee_token"] == token
        assert content["experiment_id"] is None

        assert content["ee_cert_path"] is None
        assert not (runpath / cert_file).exists()


class MockedJob:
    def __init__(self, status):
        self.queue_status = status
        self._start_time = 0
        self._current_time = 0
        self._end_time = None

    @property
    def runtime(self):
        return self._end_time - self._start_time

    def stop(self):
        self.queue_status = JobStatus.FAILED

    def convertToCReference(self, _):
        pass


def test_stop_long_running():
    """
    This test should verify that only the jobs that have a runtime
    25% longer than the average completed are stopped when
    stop_long_running_jobs is called.
    """
    job_list = [MockedJob(JobStatus.WAITING) for _ in range(10)]

    for i in range(5):
        job_list[i].queue_status = JobStatus.DONE
        job_list[i]._start_time = 0
        job_list[i]._end_time = 10

    for i in range(5, 8):
        job_list[i].queue_status = JobStatus.RUNNING
        job_list[i]._start_time = 0
        job_list[i]._end_time = 20

    for i in range(8, 10):
        job_list[i].queue_status = JobStatus.RUNNING
        job_list[i]._start_time = 0
        job_list[i]._end_time = 5

    driver = Driver(driver_type=QueueSystem.LOCAL)
    queue = JobQueue(driver)

    with patch("ert.job_queue.JobQueue._add_job") as _add_job:
        for idx, job in enumerate(job_list):
            _add_job.return_value = idx
            queue.add_job(job, idx)

    queue.stop_long_running_jobs(5)
    queue._differ.transition(queue.job_list)

    for i in range(5):
        assert job_list[i].queue_status == JobStatus.DONE
        assert queue.snapshot()[i] == str(JobStatus.DONE)

    for i in range(5, 8):
        assert job_list[i].queue_status == JobStatus.FAILED
        assert queue.snapshot()[i] == str(JobStatus.FAILED)

    for i in range(8, 10):
        assert job_list[i].queue_status == JobStatus.RUNNING
        assert queue.snapshot()[i] == str(JobStatus.RUNNING)


def test_job_queue_repr_str():
    local_driver = QueueSystem.LOCAL
    default_max_submit = 2
    repr_str = f"JobQueue({QueueSystem.LOCAL}, {default_max_submit})"
    assert repr(JobQueue(local_driver, default_max_submit)) == repr_str
    assert str(JobQueue(local_driver)) == repr_str
