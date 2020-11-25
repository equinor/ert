import pytest

from unittest.mock import Mock
from ert_shared.ensemble_evaluator.dispatch import Dispatcher
from ert_shared.ensemble_evaluator.entity import identifiers as ids


class DummyEventHandler:
    dispatch = Dispatcher()

    def __init__(self):
        self.mock_all = Mock()
        self.mock_stage = Mock()
        self.mock_none = Mock()

    @dispatch.register_event_handler(ids.EVGROUP_FM_ALL)
    async def all(self, event):
        self.mock_all(event)

    @dispatch.register_event_handler(ids.EVGROUP_FM_STAGE)
    async def stage(self, event):
        self.mock_stage(event)


def _create_dummy_event(event_type):
    return {"type": event_type}


@pytest.mark.asyncio
async def test_event_dispatcher_one_handler():
    event_handler = DummyEventHandler()

    event = _create_dummy_event(ids.EVTYPE_FM_JOB_SUCCESS)
    await DummyEventHandler.dispatch.handle_event(event_handler, event)

    event_handler.mock_all.assert_called_with(event)
    event_handler.mock_stage.assert_not_called()
    event_handler.mock_none.assert_not_called()


@pytest.mark.asyncio
async def test_event_dispatcher_two_handlers():
    event_handler = DummyEventHandler()

    event = _create_dummy_event(ids.EVTYPE_FM_STAGE_UNKNOWN)
    await DummyEventHandler.dispatch.handle_event(event_handler, event)

    event_handler.mock_all.assert_called_with(event)
    event_handler.mock_stage.assert_called_with(event)
    event_handler.mock_none.assert_not_called()


@pytest.mark.asyncio
async def test_event_dispatcher_no_handlers():
    event_handler = DummyEventHandler()

    event = _create_dummy_event("SOME_UNKNOWN_EVENT_TYPE")
    await DummyEventHandler.dispatch.handle_event(event_handler, event)

    event_handler.mock_all.assert_not_called()
    event_handler.mock_stage.assert_not_called()
    event_handler.mock_none.assert_not_called()
