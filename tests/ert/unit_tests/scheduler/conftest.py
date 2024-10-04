from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
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

            if result is None:
                return 0
            elif isinstance(result, bool):
                return 0 if result else 1
            elif isinstance(result, int):
                return result
            else:
                raise TypeError(
                    f"MockDriver's wait() function must return a bool, int or None, not {type(result)}"
                )
        return 0

    async def _kill(self, iens):
        if self._mock_kill is not None:
            if self._mock_kill.__code__.co_argcount > 0:
                await self._mock_kill(iens)
            else:
                await self._mock_kill()
        return 1


@pytest.fixture
def mock_driver():
    return MockDriver


class MockEvent(asyncio.Event):
    def __init__(self):
        self._mock_waited = asyncio.Future()
        super().__init__()

    async def wait(self) -> Coroutine[Any, Any, Literal[True]]:
        self._mock_waited.set_result(True)
        return await super().wait()

    @staticmethod
    def done() -> bool:
        return True


@pytest.fixture
def mock_event():
    return MockEvent


def mock_bin(monkeypatch, tmp_path):
    bin_path = Path(__file__).parent / "bin"

    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    monkeypatch.setenv("PYTEST_TMP_PATH", str(tmp_path))
    monkeypatch.setenv("PYTHON", sys.executable)


@pytest.fixture
def job_name(request) -> str:
    return request.node.name.split("[")[0]
