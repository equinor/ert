import asyncio

import pytest

from ert.async_utils import background_tasks


@pytest.mark.timeout(1)
async def test_background_tasks(caplog):
    current_task_future = asyncio.Future()

    async def task():
        current_task_future.set_result(asyncio.current_task())
        await asyncio.sleep(100)

    async with background_tasks() as bt:
        bt(task())
        current_task = await current_task_future
        assert not current_task.done()

    assert current_task.done()
    assert caplog.records == []


@pytest.mark.timeout(1)
async def test_background_tasks_with_exception(caplog):
    started = asyncio.Event()

    async def task():
        started.set()
        raise ValueError("Uh-oh!")

    async with background_tasks() as bt:
        bt(task())
        await started.wait()

    assert len(caplog.records) == 1
    assert caplog.records[0].message == "Uh-oh!"
