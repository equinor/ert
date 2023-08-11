from unittest.mock import Mock

import pytest

from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator.dispatch import BatchingDispatcher


class DummyEventHandler:
    def __init__(self):
        self.dispatcher = BatchingDispatcher(
            sleep_between_batches_seconds=0,
        )
        self.dispatcher._LOOKUP_MAP.clear()
        self.mock_all = Mock()
        self.mock_step = Mock()
        self.mock_none = Mock()

        self.dispatcher.register_event_handler(ids.EVGROUP_FM_ALL, self.all)
        self.dispatcher.register_event_handler(ids.EVGROUP_FM_STEP, self.step)

    async def join(self):
        await self.dispatcher.join()

    async def all(self, event):
        self.mock_all(event)

    async def step(self, event):
        self.mock_step(event)


def _create_dummy_event(event_type):
    return {"type": event_type, "source": "/ert/ee/1"}


@pytest.mark.asyncio
async def test_event_dispatcher_one_handler():
    event_handler = DummyEventHandler()

    event = _create_dummy_event(ids.EVTYPE_FM_JOB_SUCCESS)
    await event_handler.dispatcher.handle_event(event)
    await event_handler.join()

    event_handler.mock_all.assert_called_with([event])
    event_handler.mock_step.assert_not_called()
    event_handler.mock_none.assert_not_called()

    await event_handler.join()


@pytest.mark.asyncio
async def test_event_dispatcher_two_handlers():
    event_handler = DummyEventHandler()

    event = _create_dummy_event(ids.EVTYPE_FM_STEP_UNKNOWN)
    await event_handler.dispatcher.handle_event(event)
    await event_handler.join()

    event_handler.mock_all.assert_called_with([event])
    event_handler.mock_step.assert_called_with([event])
    event_handler.mock_none.assert_not_called()

    await event_handler.join()


@pytest.mark.asyncio
async def test_event_dispatcher_no_handlers():
    event_handler = DummyEventHandler()

    event = _create_dummy_event("SOME_UNKNOWN_EVENT_TYPE")
    await event_handler.dispatcher.handle_event(event)

    event_handler.mock_all.assert_not_called()
    event_handler.mock_step.assert_not_called()
    event_handler.mock_none.assert_not_called()

    await event_handler.join()


@pytest.mark.asyncio
async def test_event_dispatcher_batching_two_handlers():
    event_handler = DummyEventHandler()

    event1 = _create_dummy_event(ids.EVTYPE_FM_STEP_UNKNOWN)
    event2 = _create_dummy_event(ids.EVTYPE_FM_STEP_UNKNOWN)
    await event_handler.dispatcher.handle_event(event1)
    await event_handler.dispatcher.handle_event(event2)

    await event_handler.join()

    event_handler.mock_all.assert_called_with([event1, event2])
    event_handler.mock_step.assert_called_with([event1, event2])
    event_handler.mock_none.assert_not_called()
