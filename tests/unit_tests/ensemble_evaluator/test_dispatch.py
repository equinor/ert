from unittest.mock import Mock

import pytest

from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator.dispatch import BatchingDispatcher


class DummyEventHandler:
    """simple event handler, two different functions registered for an event group and
    an event, respectively"""

    def __init__(self):
        self.dispatcher = BatchingDispatcher(
            sleep_between_batches_seconds=0,
        )
        self.dispatcher._LOOKUP_MAP.clear()
        self.mock_fm = Mock()
        self.mock_fail = Mock()

        self.dispatcher.set_event_handler(ids.EVGROUP_FM_ALL, self.fm)
        self.dispatcher.set_event_handler({ids.EVTYPE_ENSEMBLE_FAILED}, self.fail)

    async def join(self):
        await self.dispatcher.wait_until_finished()

    def fm(self, event):
        self.mock_fm(event)

    def fail(self, event):
        self.mock_fail(event)


def _create_dummy_event(event_type):
    return {"type": event_type, "source": "/ert/ee/1"}


@pytest.mark.asyncio
async def test_that_dispatcher_uses_right_handle_function_for_one_event():
    event_handler = DummyEventHandler()

    event = _create_dummy_event(ids.EVTYPE_FORWARD_MODEL_SUCCESS)
    await event_handler.dispatcher.handle_event(event)
    await event_handler.join()

    event_handler.mock_fm.assert_called_with([event])
    event_handler.mock_fail.assert_not_called()

    await event_handler.join()


@pytest.mark.asyncio
async def test_that_event_dispatcher_uses_right_handle_functions_for_two_events():
    event_handler = DummyEventHandler()

    realization_event = _create_dummy_event(ids.EVTYPE_REALIZATION_UNKNOWN)
    fail_event = _create_dummy_event(ids.EVTYPE_ENSEMBLE_FAILED)

    await event_handler.dispatcher.handle_event(realization_event)
    await event_handler.dispatcher.handle_event(fail_event)
    await event_handler.join()

    event_handler.mock_fm.assert_called_with([realization_event])
    event_handler.mock_fail.assert_called_with([fail_event])

    await event_handler.join()


@pytest.mark.asyncio
async def test_that_event_dispatcher_ignores_event_without_registered_handle_function():
    event_handler = DummyEventHandler()

    event = _create_dummy_event("SOME_UNKNOWN_EVENT_TYPE")
    await event_handler.dispatcher.handle_event(event)

    event_handler.mock_fm.assert_not_called()
    event_handler.mock_fail.assert_not_called()

    await event_handler.join()


@pytest.mark.asyncio
async def test_event_dispatcher_batching_two_handlers():
    event_handler = DummyEventHandler()

    event1 = _create_dummy_event(ids.EVTYPE_REALIZATION_UNKNOWN)
    event2 = _create_dummy_event(ids.EVTYPE_REALIZATION_UNKNOWN)
    await event_handler.dispatcher.handle_event(event1)
    await event_handler.dispatcher.handle_event(event2)

    await event_handler.join()

    event_handler.mock_fm.assert_called_with([event1, event2])
    event_handler.mock_fail.assert_not_called()
