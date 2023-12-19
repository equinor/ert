import asyncio
import json
import shutil
from typing import Callable, List
from unittest.mock import AsyncMock, MagicMock

import pytest

import ert
from ert.ensemble_evaluator._builder._realization import Realization
from ert.load_status import LoadResult, LoadStatus
from ert.run_arg import RunArg
from ert.scheduler import Scheduler
from ert.scheduler.driver import JobEvent
from ert.scheduler.job import STATE_TO_LEGACY, Job, State


def create_scheduler():
    sch = MagicMock()
    sch._events = asyncio.Queue()
    sch.driver = AsyncMock()
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
        forward_models=[],
        active=True,
        max_runtime=None,
        run_arg=run_arg,
        num_cpu=1,
        job_script=str(shutil.which("job_dispatch.py")),
    )
    return realization


async def assert_scheduler_events(
    scheduler: Scheduler, job_events: List[JobEvent]
) -> None:
    for job_event in job_events:
        queue_event = await scheduler._events.get()
        output = json.loads(queue_event.decode("utf-8"))
        event = output.get("data").get("queue_event_type")
        assert event == STATE_TO_LEGACY[job_event.value]
    # should be no more events
    assert scheduler._events.empty()


def clear_queue(queue: asyncio.Queue) -> None:
    print(f"Clearing {queue.qsize()} items")
    while not queue.empty():
        queue.get_nowait()


@pytest.mark.asyncio
async def test_job_acquires_semaphore(realization, mock_semaphore):
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    sem: asyncio.Semaphore = mock_semaphore(value=1)
    asyncio.create_task(job._submit_and_run_once(sem))
    await asyncio.wait_for(sem._mock_locked, 5)
    assert sem.locked()


@pytest.mark.asyncio
async def test_job_waits_for_semaphore(realization, mock_semaphore, mock_event):
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    sem: asyncio.Semaphore = mock_semaphore(1)
    await sem.acquire()
    job.started = mock_event()
    asyncio.create_task(job._submit_and_run_once(sem))

    assert sem.locked()
    sem.release()
    await asyncio.wait_for(job.started._mock_waited, 5)
    await assert_scheduler_events(scheduler, [State.SUBMITTING, State.STARTING])
    scheduler.driver.submit.assert_called_once()


@pytest.mark.asyncio
async def test_job_waits_for_started_event(realization, mock_event):
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    sem = asyncio.Semaphore(1)
    job.started: asyncio.Event = mock_event()
    asyncio.create_task(job._submit_and_run_once(sem))
    await asyncio.wait_for(job.started._mock_waited, 5)
    await assert_scheduler_events(scheduler, [State.SUBMITTING, State.STARTING])

    job.started.set()
    await assert_scheduler_events(scheduler, [State.RUNNING])


@pytest.mark.asyncio
async def test_job_releases_semaphore_on_exception(realization, mock_event):
    scheduler = create_scheduler()
    scheduler.driver.submit = AsyncMock(side_effect=ZeroDivisionError)
    job = Job(scheduler, realization)
    semaphore = asyncio.BoundedSemaphore(1)

    job_task = asyncio.create_task(job._submit_and_run_once(semaphore))

    with pytest.raises(ZeroDivisionError):
        await job_task
    assert not semaphore.locked()


@pytest.mark.asyncio
async def test_job_is_cancelled(realization, mock_event, mock_semaphore):
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    sem = mock_semaphore(1)
    job.started = mock_event()
    job.aborted.set()
    job_task = asyncio.create_task(job._submit_and_run_once(sem))

    assert not job_task.cancelled()
    await asyncio.wait_for(job.started._mock_waited, 5)
    clear_queue(scheduler._events)
    assert job_task.cancel()
    await asyncio.wait_for(sem._mock_unlocked, 5)
    await assert_scheduler_events(scheduler, [State.ABORTING, State.ABORTED])
    scheduler.driver.kill.assert_called_with(job.iens)
    scheduler.driver.kill.assert_called_once()


@pytest.mark.parametrize(
    "return_code, forward_model_ok_result, expected_final_event",
    [
        [0, LoadStatus.LOAD_SUCCESSFUL, State.COMPLETED],
        [1, LoadStatus.LOAD_SUCCESSFUL, State.FAILED],
        [0, LoadStatus.LOAD_FAILURE, State.FAILED],
        [1, LoadStatus.LOAD_FAILURE, State.FAILED],
    ],
)
@pytest.mark.asyncio
async def test_job_call(
    return_code: int,
    forward_model_ok_result,
    expected_final_event: str,
    realization: Realization,
    monkeypatch,
    mock_event,
    mock_semaphore,
    mock_future: Callable[[], asyncio.Future],
):
    monkeypatch.setattr(
        ert.scheduler.job,
        "forward_model_ok",
        lambda _: LoadResult(forward_model_ok_result, ""),
    )
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    job.started = mock_event()
    job.returncode: asyncio.Future = mock_future()
    semaphore = mock_semaphore(1)

    asyncio.create_task(job._submit_and_run_once(semaphore))

    # should not be running before semaphore is available
    assert scheduler._events.empty()
    await asyncio.wait_for(job.started._mock_waited, 5)
    # should now start submitting
    await assert_scheduler_events(scheduler, [State.SUBMITTING, State.STARTING])
    scheduler.driver.submit.assert_called_with(
        realization.iens, realization.job_script, cwd=realization.run_arg.runpath
    )
    scheduler.driver.submit.assert_called_once()

    # should not run before the started event is set
    assert scheduler._events.empty()
    assert not job.started.is_set()

    # set started event
    job.started.set()
    await asyncio.wait_for(job.returncode._mock_waited, 5)
    job.returncode.set_result(return_code)
    await assert_scheduler_events(scheduler, [State.RUNNING])
    await asyncio.wait_for(semaphore._mock_unlocked, 5)
    await assert_scheduler_events(scheduler, [expected_final_event])


@pytest.mark.asyncio
async def test_job_call_waits_for_start_event(realization, mock_event):
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    sem = asyncio.Semaphore()
    start_event: asyncio.Event = mock_event()
    asyncio.create_task(job(start_event, sem))
    await asyncio.wait_for(start_event._mock_waited, timeout=5)

    assert not start_event.is_set()
    assert not sem.locked()
    await assert_scheduler_events(scheduler, [])
    scheduler.driver.submit.assert_not_called()
