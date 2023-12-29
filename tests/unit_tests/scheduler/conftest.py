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
