from __future__ import annotations

import asyncio
import contextlib
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import websockets.server

from _ert.forward_model_runner.client import Client
from _ert.threading import ErtThread
from ert.scheduler.event import FinishedEvent, StartedEvent

if TYPE_CHECKING:
    from ert.scheduler.driver import Driver


def source_dir() -> Path:
    src = Path("@CMAKE_CURRENT_SOURCE_DIR@/../..")
    if src.is_dir():
        return src.relative_to(Path.cwd())

    # If the file was not correctly configured by cmake, look for the source
    # folder, assuming the build folder is inside the source folder.
    current_path = Path(__file__)
    while current_path != Path("/"):
        if (current_path / ".git").is_dir():
            return current_path
        # This is to find root dir for git worktrees
        elif (current_path / ".git").is_file():
            with open(current_path / ".git", encoding="utf-8") as f:
                for line in f.readlines():
                    if "gitdir:" in line:
                        return current_path

        current_path = current_path.parent
    raise RuntimeError("Cannot find the source folder")


SOURCE_DIR: Path = source_dir()


def wait_until(func, interval=0.5, timeout=30):
    """Waits until func returns True.

    Repeatedly calls 'func' until it returns true.
    Waits 'interval' seconds before each invocation. If 'timeout' is
    reached, will raise the AssertionError.
    """
    t = 0
    while t < timeout:
        time.sleep(interval)
        if func():
            return
        t += interval
    raise AssertionError(
        "Timeout reached in wait_until "
        f"(function {func.__name__}, timeout {timeout:g})."
    )


def _mock_ws(host, port, messages, delay_startup=0):
    loop = asyncio.new_event_loop()
    done = loop.create_future()

    async def _handler(websocket, path):
        while True:
            msg = await websocket.recv()
            messages.append(msg)
            if msg == "stop":
                done.set_result(None)
                break

    async def _run_server():
        await asyncio.sleep(delay_startup)
        async with websockets.server.serve(_handler, host, port):
            await done

    loop.run_until_complete(_run_server())
    loop.close()


@contextlib.contextmanager
def _mock_ws_thread(host, port, messages):
    mock_ws_thread = ErtThread(
        target=partial(_mock_ws, messages=messages),
        args=(
            host,
            port,
        ),
    )
    mock_ws_thread.start()
    try:
        yield
    # Make sure to join the thread even if an exception occurs
    finally:
        url = f"ws://{host}:{port}"
        with Client(url) as client:
            client.send("stop")
        mock_ws_thread.join()
        messages.pop()


async def poll(driver: Driver, expected: set[int], *, started=None, finished=None):
    """Poll driver until expected realisations finish

    This function polls the given `driver` until realisations given by
    `expected` finish, either successfully or not, then returns. It is also
    possible to specify `started` and `finished` callbacks, for when a
    realisation starts and finishes, respectively. Blocks until all `expected`
    realisations finish.

    Parameters
    ----------
    driver : Driver
        Driver to poll
    expected : set[int]
        Set of realisation indices that we should wait for
    started : Callable[[int], None]
        Called for each job when it starts. Its associated realisation index is
        passed.
    finished : Callable[[int, int], None]
        Called for each job when it finishes. The first argument is the
        associated realisation index and the second is the returncode of the job
        process.

    """

    poll_task = asyncio.create_task(driver.poll())
    completed = set()
    try:
        while True:
            event = await driver.event_queue.get()
            if isinstance(event, StartedEvent):
                if started:
                    await started(event.iens)
            elif isinstance(event, FinishedEvent):
                if finished is not None:
                    await finished(event.iens, event.returncode)
                completed.add(event.iens)
                if completed == expected:
                    break
    finally:
        poll_task.cancel()
        await driver.finish()
