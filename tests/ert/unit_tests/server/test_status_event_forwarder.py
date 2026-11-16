import asyncio
import logging
import threading
import uuid

import pytest

from ert.ensemble_evaluator import EndEvent
from ert.run_models.event import RunModelStatusEvent, StatusEvents
from ert.server.status_event_forwarder import StatusEventForwarder
from tests.ert.utils import wait_until_all_status_events_are_delivered


def make_events(count: int) -> list[StatusEvents]:
    return [
        RunModelStatusEvent(iteration=0, run_id=uuid.uuid4(), msg=f"event {i}")
        for i in range(count)
    ]


async def test_that_listener_exception_is_logged_and_later_events_are_still_delivered(
    caplog: pytest.LogCaptureFixture,
):
    received_events: list[StatusEvents] = []
    events = make_events(2)

    def listener(*, event: StatusEvents) -> None:
        received_events.append(event)
        if event == events[0]:
            raise RuntimeError("listener failed")

    forwarder = StatusEventForwarder(listener)
    with caplog.at_level(logging.ERROR):
        for event in events:
            forwarder.forward(event)
        await wait_until_all_status_events_are_delivered(forwarder)

    assert received_events == events
    assert "Status event listener failed to handle event" in caplog.text
    assert "RuntimeError: listener failed" in caplog.text


async def test_that_blocked_listener_gets_pending_events_serially_in_order():
    listener_started = threading.Event()
    listener_may_return = threading.Event()
    lock = threading.Lock()
    received_events: list[StatusEvents] = []
    concurrent_calls = 0
    max_concurrent_calls = 0

    def listener(*, event: StatusEvents) -> None:
        nonlocal concurrent_calls, max_concurrent_calls
        with lock:
            concurrent_calls += 1
            max_concurrent_calls = max(max_concurrent_calls, concurrent_calls)
            received_events.append(event)
        listener_started.set()
        listener_may_return.wait(timeout=5)
        with lock:
            concurrent_calls -= 1

    forwarder = StatusEventForwarder(listener)
    events = [*make_events(2), EndEvent(failed=False, msg="done")]
    assert forwarder.pending == 0

    try:
        for event in events:
            forwarder.forward(event)
        assert await asyncio.to_thread(listener_started.wait, 5)

        assert forwarder.pending == len(events)
        assert received_events == events[:1]
    finally:
        listener_may_return.set()
    await wait_until_all_status_events_are_delivered(forwarder)

    assert forwarder.pending == 0
    assert received_events == events
    assert max_concurrent_calls == 1
