import asyncio
import json
import shutil
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

import ert
from ert.ensemble_evaluator._builder._realization import Realization
from ert.load_status import LoadResult, LoadStatus
from ert.run_arg import RunArg
from ert.scheduler import Scheduler
from ert.scheduler.job import STATE_TO_LEGACY, Job, State


def create_scheduler():
    sch = AsyncMock()
    sch._events = asyncio.Queue()
    sch.driver = AsyncMock()
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
        forward_models=[],
        active=True,
        max_runtime=None,
        run_arg=run_arg,
        num_cpu=1,
        job_script=str(shutil.which("job_dispatch.py")),
    )
    return realization


async def assert_scheduler_events(
    scheduler: Scheduler, expected_job_events: List[State]
) -> None:
    for job_event in expected_job_events:
        assert (
            scheduler._events.qsize()
        ), f"Expected to find {job_event=} in the event queue"
        queue_event = scheduler._events.get_nowait()
        output = json.loads(queue_event.decode("utf-8"))
        event = output.get("data").get("queue_event_type")
        assert event == STATE_TO_LEGACY[job_event]
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
        [State.WAITING, State.SUBMITTING, State.PENDING, State.ABORTING, State.ABORTED],
    )
    scheduler.driver.kill.assert_called_with(job.iens)
    scheduler.driver.kill.assert_called_once()


@pytest.mark.parametrize(
    "return_code, max_submit, forward_model_ok_result, expected_final_event",
    [
        [0, 1, LoadStatus.LOAD_SUCCESSFUL, State.COMPLETED],
        [1, 1, LoadStatus.LOAD_SUCCESSFUL, State.FAILED],
        [0, 1, LoadStatus.LOAD_FAILURE, State.FAILED],
        [1, 1, LoadStatus.LOAD_FAILURE, State.FAILED],
        [0, 2, LoadStatus.LOAD_SUCCESSFUL, State.COMPLETED],
        [1, 2, LoadStatus.LOAD_SUCCESSFUL, State.FAILED],
        [0, 2, LoadStatus.LOAD_FAILURE, State.FAILED],
        [1, 2, LoadStatus.LOAD_FAILURE, State.FAILED],
    ],
)
@pytest.mark.asyncio
async def test_job_run_sends_expected_events(
    return_code: int,
    max_submit: int,
    forward_model_ok_result,
    expected_final_event: State,
    realization: Realization,
    monkeypatch,
):
    monkeypatch.setattr(
        ert.scheduler.job,
        "forward_model_ok",
        lambda _: LoadResult(forward_model_ok_result, ""),
    )
    scheduler = create_scheduler()
    monkeypatch.setattr(
        scheduler.driver,
        "read_stdout_and_stderr_files",
        lambda *args: "",
    )

    scheduler.job.forward_model_ok = MagicMock()
    scheduler.job.forward_model_ok.return_value = LoadResult(
        forward_model_ok_result, ""
    )
    job = Job(scheduler, realization)
    job.started.set()

    job_run_task = asyncio.create_task(
        job.run(asyncio.Semaphore(), max_submit=max_submit)
    )

    for attempt in range(max_submit):
        # The execution flow through Job.run() is manipulated through job.returncode
        if attempt < max_submit - 1:
            job.returncode.set_result(1)
            while job.returncode.done():
                # wait until job.run() resets
                # the future after seeing the failure
                await asyncio.sleep(0)
        else:
            job.started.set()
            job.returncode.set_result(return_code)

    await job_run_task

    await assert_scheduler_events(
        scheduler,
        [State.WAITING, State.SUBMITTING, State.PENDING, State.RUNNING] * max_submit
        + [expected_final_event],
    )
    scheduler.driver.submit.assert_called_with(
        realization.iens,
        realization.job_script,
        realization.run_arg.runpath,
        name=realization.run_arg.job_name,
        runpath=Path(realization.run_arg.runpath),
        num_cpu=realization.num_cpu,
    )
    assert scheduler.driver.submit.call_count == max_submit
