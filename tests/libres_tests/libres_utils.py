import asyncio
import contextlib
import logging
import os
import shutil
import tempfile
import threading
import time
from functools import partial
from typing import Any, Callable

import decorator
import pytest
import websockets
from ecl.util.test import ExtendedTestCase

from ert.shared.ensemble_evaluator.client import Client

"""
Swiped from
https://github.com/equinor/everest/blob/master/tests/utils/__init__.py
"""


def tmpdir(path=None, teardown=True):
    """Decorator based on the  `tmp` context"""

    def real_decorator(function):
        def wrapper(function, *args, **kwargs):
            with tmp(path, teardown=teardown):
                return function(*args, **kwargs)

        return decorator.decorator(wrapper, function)

    return real_decorator


@contextlib.contextmanager
def tmp(path=None, teardown=True):
    """Create and go into tmp directory, returns the path.
    This function creates a temporary directory and enters that directory.  The
    returned object is the path to the created directory.
    If @path is not specified, we create an empty directory, otherwise, it must
    be a path to an existing directory.  In that case, the directory will be
    copied into the temporary directory.
    If @teardown is True (defaults to True), the directory is (attempted)
    deleted after context, otherwise it is kept as is.
    """
    cwd = os.getcwd()
    fname = tempfile.NamedTemporaryFile().name

    if path:
        if not os.path.isdir(path):
            logging.debug("tmp:raise no such path")
            raise IOError(f"No such directory: {path}")
        shutil.copytree(path, fname)
    else:
        # no path to copy, create empty dir
        os.mkdir(fname)

    os.chdir(fname)

    yield fname  # give control to caller scope

    os.chdir(cwd)

    if teardown:
        try:
            shutil.rmtree(fname)
        except OSError as oserr:
            logging.debug(f"tmp:rmtree failed {fname} ({oserr})")
            shutil.rmtree(fname, ignore_errors=True)


def wait_for(
    func: Callable, target: Any = True, interval: float = 0.1, timeout: float = 30
):
    """Sleeps (with timeout) until the provided function returns the provided target"""
    t = 0.0
    while func() != target:
        time.sleep(interval)
        t += interval
        if t >= timeout:
            raise AssertionError(
                "Timeout reached in wait_for "
                f"(function {func.__name__}, timeout {timeout}) "
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
        async with websockets.serve(_handler, host, port):
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
    yield
    url = f"ws://{host}:{port}"
    with Client(url) as client:
        client.send("stop")
    mock_ws_thread.join()
    messages.pop()


@pytest.mark.usefixtures("class_source_root")
class ResTest(ExtendedTestCase):
    pass
