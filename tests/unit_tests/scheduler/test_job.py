import asyncio
import shutil
from typing import List
from unittest.mock import MagicMock

import pytest

import ert.scheduler.job
from ert.ensemble_evaluator._builder._realization import Realization
from ert.load_status import LoadResult, LoadStatus
from ert.run_arg import RunArg
from ert.scheduler.event_sender import EventSender
from ert.scheduler.job import STATE_TO_LEGACY, Job, State


@pytest.fixture
def driver(mock_driver):
    return mock_driver()


@pytest.fixture
def event_sender():
    return EventSender(None, None, None, None)


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


async def assert_events(event_sender: EventSender, job_events: List[State]) -> None:
    for job_event in job_events:
        queue_event = await event_sender.events.get()
        event = (queue_event.get_data() or {}).get("queue_event_type")
        assert event == STATE_TO_LEGACY[job_event]
    # should be no more events
    assert event_sender.events.empty()


@pytest.mark.timeout(5)
async def test_submitted_job_is_cancelled(
    realization, mock_event, event_sender, driver
):
    job = Job(realization)
    job.started = mock_event()
    job.returncode.cancel()
    job_task = asyncio.create_task(
        job(asyncio.BoundedSemaphore(), event_sender, driver, max_submit=1)
    )

    await asyncio.wait_for(job.started._mock_waited, 5)

    assert job_task.cancel()
    await job_task
    await assert_events(event_sender, [State.SUBMITTING, State.PENDING, State.ABORTED])


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
    expected_final_event: State,
    realization: Realization,
    event_sender,
    driver,
    monkeypatch,
):
    monkeypatch.setattr(
        ert.scheduler.job,
        "forward_model_ok",
        lambda _: LoadResult(forward_model_ok_result, ""),
    )
    job = Job(realization)
    job.started.set()
    job.returncode.set_result(return_code)

    await job(asyncio.BoundedSemaphore(), event_sender, driver, max_submit=1)

    await assert_events(
        event_sender,
        [State.SUBMITTING, State.PENDING, State.RUNNING, expected_final_event],
    )
