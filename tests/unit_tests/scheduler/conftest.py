import asyncio
from collections.abc import Generator
from typing import Any, Coroutine, Literal

import pytest

from ert.scheduler.local_driver import LocalDriver


class MockDriver(LocalDriver):
    def __init__(self, init=None, wait=None, kill=None):
        super().__init__()
        self._mock_init = init
        self._mock_wait = wait
        self._mock_kill = kill

    async def _init(self, iens, *args, **kwargs):
        if self._mock_init is not None:
            await self._mock_init(iens, *args, **kwargs)
        return iens

    async def _wait(self, iens):
        if self._mock_wait is not None:
            if self._mock_wait.__code__.co_argcount > 0:
                result = await self._mock_wait(iens)
            else:
                result = await self._mock_wait()
            return True if result is None else bool(result)
        return True

    async def _kill(self, iens, *args):
        if self._mock_kill is not None:
            if self._mock_kill.__code__.co_argcount > 0:
                await self._mock_kill(iens)
            else:
                await self._mock_kill()


@pytest.fixture
def mock_driver():
    return MockDriver


class MockSemaphore(asyncio.Semaphore):
    def __init__(self, value: int):
        super().__init__(value)
        self._mock_locked = asyncio.Future()
        self._mock_unlocked = asyncio.Future()

    async def acquire(self) -> Coroutine[Any, Any, Literal[True]]:
        if self._mock_locked.done():
            self._mock_locked = asyncio.Future()
        self._mock_locked.set_result(True)
        return await super().acquire()

    def release(self) -> None:
        if self._mock_unlocked.done():
            self._mock_unlocked = asyncio.Future()
        self._mock_unlocked.set_result(True)
        return super().release()


@pytest.fixture
def mock_semaphore():
    return MockSemaphore


class MockEvent(asyncio.Event):
    def __init__(self):
        self._mock_waited = asyncio.Future()
        super().__init__()

    async def wait(self) -> Coroutine[Any, Any, Literal[True]]:
        self._mock_waited.set_result(True)
        return await super().wait()

    def done(self) -> bool:
        return True


@pytest.fixture
def mock_event():
    return MockEvent


class MockFuture(asyncio.Future):
    def __init__(self):
        super().__init__()
        self._done = False
        self._mock_waited = asyncio.Future()

    def done(self) -> bool:
        if self._done:
            return True
        self._done = True
        return False

    def __await__(self) -> Generator[Any, None, Any]:
        self._mock_waited.set_result(True)
        return super().__await__()


@pytest.fixture
def mock_future():
    return MockFuture
