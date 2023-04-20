import asyncio
import contextlib
import threading
import time
from functools import partial
from pathlib import Path

import websockets.server

from _ert_job_runner.client import Client


def source_dir():
    src = Path("@CMAKE_CURRENT_SOURCE_DIR@/../..")
    if src.is_dir():
        return src.relative_to(Path.cwd())

    # If the file was not correctly configured by cmake, look for the source
    # folder, assuming the build folder is inside the source folder.
    current_path = Path(__file__)
    while current_path != Path("/"):
        if (current_path / ".git").is_dir():
            return current_path
        current_path = current_path.parent
    raise RuntimeError("Cannot find the source folder")


SOURCE_DIR = source_dir()


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
        async with websockets.server.serve(_handler, host, port, open_timeout=10):
            await done

    loop.run_until_complete(_run_server())
    loop.close()


@contextlib.contextmanager
def _mock_ws_thread(host, port, messages):
    mock_ws_thread = threading.Thread(
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
