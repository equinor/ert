import asyncio
import json
import shutil
from unittest.mock import AsyncMock, MagicMock
from typing import List
import pytest

import ert
from ert.ensemble_evaluator._builder._realization import Realization
from ert.load_status import LoadResult, LoadStatus
from ert.run_arg import RunArg
from ert.scheduler import Scheduler
from ert.scheduler.driver import JobEvent
from ert.scheduler.job import STATE_TO_LEGACY, Job, State


@pytest.fixture
def scheduler():
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
    await asyncio.sleep(0)
    assert scheduler._events.qsize() == len(job_events)
    for job_event in job_events:
        output = json.loads((await scheduler._events.get()).decode("utf-8"))
        event = output.get("data").get("queue_event_type")
        assert event == STATE_TO_LEGACY[job_event.value]
    # should be no more events
    assert scheduler._events.empty()
    await asyncio.sleep(0)


def clear_queue(queue: asyncio.Queue) -> None:
    print(f"Clearing {queue.qsize()} items")
    while not queue.empty():
        queue.get_nowait()


@pytest.mark.asyncio
async def test_job_acquires_semaphore(scheduler, realization):
    job = Job(scheduler, realization)
    sem = asyncio.Semaphore(1)
    start_event = asyncio.Event()
    asyncio.create_task(job(start_event, sem))
    await asyncio.sleep(0)
    start_event.set()
    await asyncio.sleep(0)
    assert sem.locked()


@pytest.mark.asyncio
async def test_job_waits_for_start_event(scheduler, realization):
    job = Job(scheduler, realization)
    sem = asyncio.Semaphore(1)
    start_event = asyncio.Event()
    asyncio.create_task(job(start_event, sem))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert not start_event.is_set()
    assert not sem.locked()
    await assert_scheduler_events(scheduler, [])
    scheduler.driver.submit.assert_not_called()


@pytest.mark.asyncio
async def test_job_waits_for_semaphore(scheduler, realization):
    job = Job(scheduler, realization)
    sem = asyncio.Semaphore(1)
    await sem.acquire()

    start_event = asyncio.Event()
    asyncio.create_task(job(start_event, sem))
    start_event.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert sem.locked()
    await assert_scheduler_events(scheduler, [])
    sem.release()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await assert_scheduler_events(scheduler, [State.SUBMITTING, State.STARTING])
    scheduler.driver.submit.assert_called_once()


@pytest.mark.asyncio
async def test_job_waits_for_started_event(scheduler, realization):
    job = Job(scheduler, realization)
    sem = asyncio.Semaphore(1)

    start_event = asyncio.Event()
    asyncio.create_task(job(start_event, sem))
    start_event.set()

    await assert_scheduler_events(scheduler, [State.SUBMITTING, State.STARTING])

    job.started.set()
    await assert_scheduler_events(scheduler, [State.RUNNING])


@pytest.mark.asyncio
async def test_job_releases_semaphore_on_exception(scheduler, realization):
    scheduler.driver.submit = AsyncMock(side_effect=ZeroDivisionError)
    job = Job(scheduler, realization)
    start_event = asyncio.Event()
    semaphore = asyncio.BoundedSemaphore(1)

    job_task = asyncio.create_task(job(start_event, semaphore))
    start_event.set()
    # should now start submitting
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    with pytest.raises(ZeroDivisionError):
        job_task.result()
    assert not semaphore.locked()


@pytest.mark.asyncio
async def test_job_is_cancelled(scheduler: Scheduler, realization):
    job = Job(scheduler, realization)
    sem = asyncio.Semaphore(1)

    start_event = asyncio.Event()
    job_task = asyncio.create_task(job(start_event, sem))
    start_event.set()
    await asyncio.sleep(0)
    assert not job_task.cancelled()
    clear_queue(scheduler._events)
    assert job_task.cancel()
    await assert_scheduler_events(scheduler, [State.ABORTING])
    scheduler.driver.kill.assert_called_with(job.iens)
    scheduler.driver.kill.assert_called_once()
    job.aborted.set()
    await assert_scheduler_events(scheduler, [State.ABORTED])


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
    scheduler: Scheduler,
    realization: Realization,
    monkeypatch,
):
    monkeypatch.setattr(
        ert.scheduler.job,
        "forward_model_ok",
        lambda _: LoadResult(forward_model_ok_result, ""),
    )

    job = Job(scheduler, realization)

    start_event = asyncio.Event()
    semaphore = asyncio.BoundedSemaphore(1)

    asyncio.create_task(job(start_event, semaphore))
    start_event.set()

    # should not be running before semaphore is available
    assert scheduler._events.empty()

    # should now start submitting
    # SHOULD SPLIT THIS TO MAKE SURE State.Starting is ran AFTER scheduler.driver.submit()
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
    await assert_scheduler_events(scheduler, [State.RUNNING])
    job.returncode.set_result(return_code)
    await asyncio.sleep(0)
    await assert_scheduler_events(scheduler, [expected_final_event])

    # should release semaphore regardless of ending
    assert not semaphore.locked()
