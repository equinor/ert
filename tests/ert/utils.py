from __future__ import annotations

import asyncio
import contextlib
import queue
import time
from pathlib import Path
from typing import TYPE_CHECKING

import zmq
import zmq.asyncio

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


async def async_mock_zmq_server(messages, port, server_started):
    async def _handler(router_socket):
        while True:
            dealer, __, frame = await router_socket.recv_multipart()
            await router_socket.send_multipart([dealer, b"", b"ACK"])
            raw_msg = frame.decode("utf-8")
            messages.append(raw_msg)
            if raw_msg == "DISCONNECT":
                return

    zmq_context = zmq.asyncio.Context()  # type: ignore
    router_socket = zmq_context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://*:{port}")
    server_started.set()
    await _handler(router_socket)
    router_socket.close()
    zmq_context.term()


@contextlib.contextmanager
def mock_zmq_thread(port, messages, signal_queue=None):
    loop = None
    handler_task = None

    def mock_zmq_server(messages, port, signal_queue=None):
        nonlocal loop, handler_task
        loop = asyncio.new_event_loop()

        async def _handler(router_socket):
            nonlocal messages, signal_queue
            signal_value = 0
            while True:
                try:
                    dealer, __, frame = await router_socket.recv_multipart()
                    if signal_queue:
                        with contextlib.suppress(queue.Empty):
                            signal_value = signal_queue.get(timeout=0.1)

                    print(f"{dealer=} {frame=} {signal_value=}")
                    if frame in [b"CONNECT", b"DISCONNECT"] or signal_value == 0:
                        await router_socket.send_multipart([dealer, b"", b"ACK"])
                    if frame not in [b"CONNECT", b"DISCONNECT"] and signal_value != 1:
                        messages.append(frame.decode("utf-8"))

                except asyncio.CancelledError:
                    break

        async def _run_server():
            nonlocal handler_task
            zmq_context = zmq.asyncio.Context()  # type: ignore
            router_socket = zmq_context.socket(zmq.ROUTER)
            router_socket.bind(f"tcp://*:{port}")
            handler_task = asyncio.create_task(_handler(router_socket))
            await handler_task
            router_socket.close()

        loop.run_until_complete(_run_server())
        loop.close()

    mock_zmq_thread = ErtThread(
        target=lambda: mock_zmq_server(messages, port, signal_queue),
    )
    mock_zmq_thread.start()
    try:
        yield
    finally:
        print(f"these are the final {messages=}")
        if handler_task and not handler_task.done():
            loop.call_soon_threadsafe(handler_task.cancel)
        mock_zmq_thread.join()


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
