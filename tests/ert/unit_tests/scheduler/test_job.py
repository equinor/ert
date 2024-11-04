import asyncio
import logging
import shutil
from functools import partial
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from lxml import etree

import ert
from ert.ensemble_evaluator import Realization
from ert.load_status import LoadStatus
from ert.run_arg import RunArg
from ert.run_models.base_run_model import captured_logs
from ert.scheduler import Scheduler
from ert.scheduler.job import (
    Job,
    JobState,
    log_info_from_exit_file,
)


def create_scheduler():
    sch = AsyncMock()
    sch._ens_id = "0"
    sch._events = asyncio.Queue()
    sch.driver = AsyncMock()
    sch._manifest_queue = None
    sch._cancelled = False
    return sch


@pytest.fixture
def realization():
    run_arg = RunArg(
        run_id="",
        ensemble_storage=MagicMock(),
        iens=0,
        itr=0,
        runpath="test_runpath",
        job_name="test_job",
    )
    realization = Realization(
        iens=run_arg.iens,
        fm_steps=[],
        active=True,
        max_runtime=None,
        run_arg=run_arg,
        num_cpu=1,
        job_script=str(shutil.which("job_dispatch.py")),
        realization_memory=0,
    )
    return realization


async def assert_scheduler_events(
    scheduler: Scheduler, expected_job_events: List[JobState]
) -> None:
    for expected_job_event in expected_job_events:
        assert (
            scheduler._events.qsize()
        ), f"Expected to find {expected_job_event=} in the event queue"
        event = scheduler._events.get_nowait()
        assert event.queue_event_type == expected_job_event

    # should be no more events
    assert scheduler._events.empty()


@pytest.mark.timeout(5)
async def test_submitted_job_is_cancelled(realization, mock_event):
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    job._requested_max_submit = 1
    job.started = mock_event()
    job.returncode.cancel()
    job_task = asyncio.create_task(job._submit_and_run_once(asyncio.BoundedSemaphore()))

    await asyncio.wait_for(job.started._mock_waited, 5)

    job_task.cancel()
    await job_task

    await assert_scheduler_events(
        scheduler,
        [
            JobState.WAITING,
            JobState.SUBMITTING,
            JobState.PENDING,
            JobState.ABORTING,
            JobState.ABORTED,
        ],
    )
    scheduler.driver.kill.assert_called_with(job.iens)
    scheduler.driver.kill.assert_called_once()


@pytest.mark.parametrize(
    "return_code, max_submit, forward_model_ok_result, expected_final_event",
    [
        [0, 1, LoadStatus.LOAD_SUCCESSFUL, JobState.COMPLETED],
        [1, 1, LoadStatus.LOAD_SUCCESSFUL, JobState.FAILED],
        [0, 1, LoadStatus.LOAD_FAILURE, JobState.FAILED],
        [1, 1, LoadStatus.LOAD_FAILURE, JobState.FAILED],
        [0, 2, LoadStatus.LOAD_SUCCESSFUL, JobState.COMPLETED],
        [1, 2, LoadStatus.LOAD_SUCCESSFUL, JobState.FAILED],
        [0, 2, LoadStatus.LOAD_FAILURE, JobState.FAILED],
        [1, 2, LoadStatus.LOAD_FAILURE, JobState.FAILED],
    ],
)
@pytest.mark.asyncio
async def test_job_run_sends_expected_events(
    return_code: int,
    max_submit: int,
    forward_model_ok_result,
    expected_final_event: JobState,
    realization: Realization,
    monkeypatch,
):
    async def load_result(**_):
        return (forward_model_ok_result, "")

    monkeypatch.setattr(ert.scheduler.job, "forward_model_ok", load_result)
    scheduler = create_scheduler()
    monkeypatch.setattr(
        scheduler.driver,
        "read_stdout_and_stderr_files",
        lambda *args: "",
    )

    job = Job(scheduler, realization)
    job._verify_checksum = partial(job._verify_checksum, timeout=0)
    job.started.set()

    job_run_task = asyncio.create_task(
        job.run(
            asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=max_submit
        )
    )

    for attempt in range(max_submit):
        # The execution flow through Job.run() is manipulated through job.returncode
        if attempt < max_submit - 1:
            job.returncode.set_result(1)
            while job.returncode.done():  # noqa: ASYNC110
                # wait until job.run() resets
                # the future after seeing the failure
                await asyncio.sleep(0)
        else:
            job.started.set()
            job.returncode.set_result(return_code)

    await job_run_task

    await assert_scheduler_events(
        scheduler,
        [JobState.WAITING, JobState.SUBMITTING, JobState.PENDING, JobState.RUNNING]
        * max_submit
        + [expected_final_event],
    )
    scheduler.driver.submit.assert_called_with(
        realization.iens,
        realization.job_script,
        realization.run_arg.runpath,
        name=realization.run_arg.job_name,
        runpath=Path(realization.run_arg.runpath),
        num_cpu=realization.num_cpu,
        realization_memory=0,
    )
    assert scheduler.driver.submit.call_count == max_submit


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.asyncio
async def test_num_cpu_is_propagated_to_driver(realization: Realization):
    realization.num_cpu = 8
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    job_run_task = asyncio.create_task(
        job.run(asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=1)
    )
    job.started.set()
    job.returncode.set_result(0)
    await job_run_task
    scheduler.driver.submit.assert_called_with(
        realization.iens,
        realization.job_script,
        realization.run_arg.runpath,
        name=realization.run_arg.job_name,
        runpath=Path(realization.run_arg.runpath),
        num_cpu=8,
        realization_memory=0,
    )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.asyncio
async def test_realization_memory_is_propagated_to_driver(realization: Realization):
    realization.realization_memory = 8 * 1024**2
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    job_run_task = asyncio.create_task(
        job.run(asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=1)
    )
    job.started.set()
    job.returncode.set_result(0)
    await job_run_task
    scheduler.driver.submit.assert_called_with(
        realization.iens,
        realization.job_script,
        realization.run_arg.runpath,
        num_cpu=1,
        realization_memory=8 * 1024**2,
        name=realization.run_arg.job_name,
        runpath=Path(realization.run_arg.runpath),
    )


@pytest.mark.asyncio
async def test_when_waiting_for_disk_sync_times_out_an_error_is_logged(
    realization: Realization, monkeypatch
):
    scheduler = create_scheduler()
    scheduler._manifest_queue = asyncio.Queue()
    file_path = "does/not/exist"
    scheduler.checksum = {
        "test_runpath": {
            "file": {
                "path": file_path,
                "md5sum": "something",
            }
        }
    }
    log_msgs = []
    job = Job(scheduler, realization)
    job._verify_checksum = partial(job._verify_checksum, timeout=0)
    job.started.set()

    with captured_logs(log_msgs, logging.ERROR):
        job_run_task = asyncio.create_task(
            job.run(asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=1)
        )
        job.started.set()
        job.returncode.set_result(0)
        await job_run_task

    assert "Disk synchronization failed for does/not/exist" in log_msgs


@pytest.mark.asyncio
async def test_when_files_in_manifest_are_not_created_an_error_is_logged(
    realization: Realization, monkeypatch
):
    scheduler = create_scheduler()
    scheduler._manifest_queue = asyncio.Queue()
    file_path = "does/not/exist"
    error = f"Expected file {file_path} not created by forward model!"
    scheduler.checksum = {
        "test_runpath": {
            "file": {
                "path": file_path,
                "error": error,
            }
        }
    }
    log_msgs = []
    job = Job(scheduler, realization)
    job.started.set()

    with captured_logs(log_msgs, logging.ERROR):
        job_run_task = asyncio.create_task(
            job.run(asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=1)
        )
        job.started.set()
        job.returncode.set_result(0)
        await job_run_task

    assert error in log_msgs


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.asyncio
async def test_when_checksums_do_not_match_a_warning_is_logged(
    realization: Realization,
):
    scheduler = create_scheduler()
    scheduler._manifest_queue = asyncio.Queue()
    file_path = "invalid_md5sum"
    scheduler.checksum = {
        "test_runpath": {
            "file": {
                "path": file_path,
                "md5sum": "something_something_checksum",
            }
        }
    }
    # Create the file
    Path(file_path).write_text("test", encoding="utf-8")

    log_msgs = []
    job = Job(scheduler, realization)
    job.started.set()

    with captured_logs(log_msgs, logging.WARNING):
        job_run_task = asyncio.create_task(
            job.run(asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=1)
        )
        job.started.set()
        job.returncode.set_result(0)
        await job_run_task

    assert f"File {file_path} checksum verification failed." in log_msgs


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.asyncio
async def test_when_no_checksum_info_is_received_a_warning_is_logged(
    realization: Realization, mocker
):
    scheduler = create_scheduler()
    scheduler._manifest_queue = asyncio.Queue()
    scheduler.checksum = {}
    # Create the file

    log_msgs = []
    job = Job(scheduler, realization)
    job.started.set()

    # Mock asyncio.sleep to fast-forward time
    mocker.patch("asyncio.sleep", return_value=None)

    with captured_logs(log_msgs, logging.WARNING):
        job_run_task = asyncio.create_task(
            job.run(asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=1)
        )
        job.started.set()
        job.returncode.set_result(0)
        await job_run_task

    assert (
        f"Checksum information not received for {realization.run_arg.runpath}"
        in log_msgs
    )


@pytest.mark.usefixtures("use_tmpdir")
async def test_log_info_from_exit_file(caplog):
    exit_contents = {
        "job": "foojob",
        "reason": "Divizion-by-sero",
        "stderr_file": "somefilename",
        "stderr": "some_error",
    }

    root = etree.Element("root")
    for key, value in exit_contents.items():
        node = etree.Element(key)
        node.text = value
        root.append(node)

    Path("ERROR").write_text(
        str(etree.tostring(root), encoding="utf-8", errors="ignore"), encoding="utf-8"
    )

    log_info_from_exit_file(Path("ERROR"))
    logs = caplog.text
    for magic_string in exit_contents.values():
        assert magic_string in logs


@pytest.mark.usefixtures("use_tmpdir")
async def test_log_info_from_garbled_exit_file(caplog):
    Path("ERROR").write_text("this is not XML", encoding="utf-8")
    log_info_from_exit_file(Path("ERROR"))
    logs = caplog.text
    assert "job failed with invalid XML" in logs
    assert "'this is not XML'" in logs
