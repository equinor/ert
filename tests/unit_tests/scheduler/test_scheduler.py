import asyncio
import json
import shutil
from pathlib import Path
from typing import Sequence

import pytest

from ert.ensemble_evaluator._builder._realization import Realization
from ert.job_queue.queue import EVTYPE_ENSEMBLE_STOPPED
from ert.run_arg import RunArg
from ert.scheduler import scheduler


@pytest.fixture
def realization(storage, tmp_path):
    ensemble = storage.create_experiment().create_ensemble(name="foo", ensemble_size=1)

    run_arg = RunArg(
        run_id="",
        ensemble_storage=ensemble,
        iens=0,
        itr=0,
        runpath=str(tmp_path),
        job_name="",
    )

    return Realization(
        iens=0,
        forward_models=[],
        active=True,
        max_runtime=None,
        run_arg=run_arg,
        num_cpu=1,
        job_script=str(shutil.which("job_dispatch.py")),
    )


async def test_empty():
    sch = scheduler.Scheduler()
    assert await sch.execute() == EVTYPE_ENSEMBLE_STOPPED


async def test_single_job(realization, mock_driver):
    future = asyncio.Future()

    async def init(iens, *args, **kwargs):
        future.set_result(iens)

    driver = mock_driver(init=init)

    sch = scheduler.Scheduler(driver)
    sch.add_realization(realization)

    assert await sch.execute() == EVTYPE_ENSEMBLE_STOPPED
    assert await future == realization.iens


async def test_cancel(realization, mock_driver):
    pre = asyncio.Event()
    post = asyncio.Event()
    killed = False

    async def wait():
        pre.set()
        await asyncio.sleep(10)
        post.set()

    async def kill():
        nonlocal killed
        killed = True

    driver = mock_driver(wait=wait, kill=kill)
    sch = scheduler.Scheduler(driver)
    sch.add_realization(realization)

    scheduler_task = asyncio.create_task(sch.execute())

    # Wait for the job to start
    await asyncio.wait_for(pre.wait(), timeout=1)

    # Kill all jobs and wait for the scheduler to complete
    sch.kill_all_jobs()
    await scheduler_task

    assert pre.is_set()
    assert not post.is_set()
    assert killed


@pytest.mark.parametrize(
    "max_submit",
    [
        (1),
        (2),
        (3),
    ],
)
async def test_that_max_submit_was_reached(realization, max_submit, mock_driver):
    retries = 0

    async def init(*args, **kwargs):
        nonlocal retries
        retries += 1

    async def wait():
        return False

    driver = mock_driver(init=init, wait=wait)
    sch = scheduler.Scheduler(driver)

    sch._max_submit = max_submit
    sch.add_realization(realization, callback_timeout=lambda _: None)

    assert await sch.execute() == EVTYPE_ENSEMBLE_STOPPED
    assert retries == max_submit
