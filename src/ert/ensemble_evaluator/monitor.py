from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Final

from _ert.events import (
    EETerminated,
    EEUserCancel,
    EEUserDone,
    Event,
    event_from_json,
    event_to_json,
)
from _ert.forward_model_runner.client import Client

logger = logging.getLogger(__name__)


class EventSentinel:
    pass


class Monitor(Client):
    _sentinel: Final = EventSentinel()

    def __init__(self, uri: str, token: str | None = None) -> None:
        self._id = str(uuid.uuid1()).split("-", maxsplit=1)[0]
        self._event_queue: asyncio.Queue[Event | EventSentinel] = asyncio.Queue()
        self._receiver_timeout: float = 60.0
        super().__init__(uri, token, dealer_name=f"client-{self._id}")

    async def process_message(self, msg: str) -> None:
        event = event_from_json(msg)
        await self._event_queue.put(event)

    async def signal_cancel(self) -> None:
        await self._event_queue.put(Monitor._sentinel)
        logger.debug(f"monitor-{self._id} asking server to cancel...")
        cancel_event = EEUserCancel(monitor=self._id)
        await self.send(event_to_json(cancel_event))
        logger.debug(f"monitor-{self._id} asked server to cancel")

    async def signal_done(self) -> None:
        await self._event_queue.put(Monitor._sentinel)
        logger.debug(f"monitor-{self._id} informing server monitor is done...")

        done_event = EEUserDone(monitor=self._id)
        await self.send(event_to_json(done_event))
        logger.debug(f"monitor-{self._id} informed server monitor is done")

    async def track(
        self, heartbeat_interval: float | None = None
    ) -> AsyncGenerator[Event | None, None]:
        """Yield events from the internal event queue with optional heartbeats.

        Heartbeats are represented by None being yielded.

        Heartbeats stops being emitted after a CloseTrackerEvent is found."""
        heartbeat_interval_: float | None = heartbeat_interval
        closetracker_received: bool = False
        while True:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=heartbeat_interval_
                )
            except TimeoutError:
                if closetracker_received:
                    logger.error("Evaluator did not send the TERMINATED event!")
                    break
                event = None
            if isinstance(event, EventSentinel):
                closetracker_received = True
                heartbeat_interval_ = self._receiver_timeout
            else:
                yield event
                if type(event) is EETerminated:
                    logger.debug(f"monitor-{self._id} client received terminated")
                    break
            if event is not None:
                self._event_queue.task_done()
