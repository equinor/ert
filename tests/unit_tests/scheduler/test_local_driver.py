import asyncio

import pytest

from ert.scheduler import local_driver
from ert.scheduler.driver import JobEvent
from ert.scheduler.local_driver import LocalDriver


async def test_success(tmp_path):
    driver = LocalDriver()

    await driver.submit(42, "/usr/bin/env", "touch", "testfile", cwd=tmp_path)
    assert await driver.event_queue.get() == (42, JobEvent.STARTED)
    assert await driver.event_queue.get() == (42, JobEvent.COMPLETED)

    assert (tmp_path / "testfile").exists()


async def test_failure(tmp_path):
    driver = LocalDriver()

    await driver.submit(42, "/usr/bin/env", "false", cwd=tmp_path)
    assert await driver.event_queue.get() == (42, JobEvent.STARTED)
    assert await driver.event_queue.get() == (42, JobEvent.FAILED)


async def test_file_not_found(tmp_path):
    driver = LocalDriver()

    await driver.submit(42, "/file/not/found", cwd=tmp_path)
    assert await driver.event_queue.get() == (42, JobEvent.FAILED)


async def test_kill(tmp_path):
    driver = LocalDriver()

    await driver.submit(42, "/usr/bin/env", "sleep", "10", cwd=tmp_path)
    assert await driver.event_queue.get() == (42, JobEvent.STARTED)
    await driver.kill(42)
    assert await driver.event_queue.get() == (42, JobEvent.ABORTED)


@pytest.mark.timeout(5)
async def test_kill_unresponsive_process(monkeypatch, tmp_path):
    # Reduce timeout to something more appropriate for a test
    monkeypatch.setattr(local_driver, "_TERMINATE_TIMEOUT", 0.1)

    (tmp_path / "script").write_text(
        """\
    trap "" 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    sleep 60
    """
    )

    driver = LocalDriver()

    await driver.submit(42, "/bin/sh", tmp_path / "script", cwd=tmp_path)
    assert await driver.event_queue.get() == (42, JobEvent.STARTED)
    await driver.kill(42)
    assert await driver.event_queue.get() == (42, JobEvent.ABORTED)


@pytest.mark.parametrize(
    "cmd,event", [("true", JobEvent.COMPLETED), ("false", JobEvent.FAILED)]
)
async def test_kill_when_job_completed(tmp_path, cmd, event):
    driver = LocalDriver()

    await driver.submit(42, "/usr/bin/env", cmd, cwd=tmp_path)
    assert await driver.event_queue.get() == (42, JobEvent.STARTED)
    await asyncio.sleep(0.5)
    await driver.kill(42)
    assert await driver.event_queue.get() == (42, event)
