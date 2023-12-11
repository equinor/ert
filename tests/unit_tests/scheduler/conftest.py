import pytest

from ert.scheduler.local_driver import LocalDriver


class MockDriver(LocalDriver):
    def __init__(self, init=None, wait=None, kill=None):
        super().__init__()
        self._mock_init = init
        self._mock_wait = wait
        self._mock_kill = kill

    async def _init(self, *args, **kwargs):
        if self._mock_init is not None:
            await self._mock_init(*args, **kwargs)

    async def _wait(self, *args):
        if self._mock_wait is not None:
            result = await self._mock_wait()
            return True if result is None else bool(result)
        return True

    async def _kill(self, *args):
        if self._mock_kill is not None:
            await self._mock_kill()


@pytest.fixture
def mock_driver():
    return MockDriver
