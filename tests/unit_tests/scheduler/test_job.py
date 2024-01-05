import asyncio
import json
import shutil
from typing import List
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
    sch = AsyncMock()
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


@pytest.mark.asyncio
async def test_submitted_job_is_cancelled(realization, mock_event):
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    job._requested_max_submit = 1
    job.started = mock_event()
    job.aborted.set()
    job_task = asyncio.create_task(job._submit_and_run_once(asyncio.Semaphore()))

    await asyncio.wait_for(job.started._mock_waited, 5)

    assert job_task.cancel()
    await job_task
    await assert_scheduler_events(
        scheduler, [State.SUBMITTING, State.PENDING, State.ABORTING, State.ABORTED]
    )
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
async def test_job_submit_and_run_once(
    return_code: int,
    forward_model_ok_result,
    expected_final_event: str,
    realization: Realization,
    monkeypatch,
):
    monkeypatch.setattr(
        ert.scheduler.job,
        "forward_model_ok",
        lambda _: LoadResult(forward_model_ok_result, ""),
    )
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    job._requested_max_submit = 1
    job.started.set()
    job.returncode.set_result(return_code)

    await job._submit_and_run_once(asyncio.Semaphore())

    await assert_scheduler_events(
        scheduler,
        [State.SUBMITTING, State.PENDING, State.RUNNING, expected_final_event],
    )
    scheduler.driver.submit.assert_called_with(
        realization.iens, realization.job_script, cwd=realization.run_arg.runpath
    )
    scheduler.driver.submit.assert_called_once()
