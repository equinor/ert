import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, AsyncGenerator, Final, Optional

from _ert.events import (
    EETerminated,
    EEUserCancel,
    EEUserDone,
    Event,
)

if TYPE_CHECKING:
    from ert.ensemble_evaluator.evaluator_connection_info import EvaluatorConnectionInfo


logger = logging.getLogger(__name__)


class EventSentinel:
    pass


class Monitor:
    _sentinel: Final = EventSentinel()

    def __init__(
        self,
        ee_con_info: "EvaluatorConnectionInfo",
        monitor_to_ee_queue: asyncio.Queue,
        ee_to_monitor_queue: asyncio.Queue,
    ) -> None:
        self._ee_con_info = ee_con_info
        self._id = str(uuid.uuid1()).split("-", maxsplit=1)[0]
        self._connected: asyncio.Event = asyncio.Event()
        self._connection_timeout: float = 120.0
        self._receiver_timeout: float = 60.0
        self._monitor_to_ee_queue = monitor_to_ee_queue
        self._ee_to_monitor_queue = ee_to_monitor_queue

    async def signal_cancel(self) -> None:
        await self._ee_to_monitor_queue.put(Monitor._sentinel)
        logger.debug(f"monitor-{self._id} asking server to cancel...")

        cancel_event = EEUserCancel(monitor=self._id)
        await self._monitor_to_ee_queue.put(cancel_event)
        logger.debug(f"monitor-{self._id} asked server to cancel")

    async def signal_done(self) -> None:
        await self._ee_to_monitor_queue.put(Monitor._sentinel)
        logger.debug(f"monitor-{self._id} informing server monitor is done...")

        done_event = EEUserDone(monitor=self._id)
        await self._monitor_to_ee_queue.put(done_event)
        logger.debug(f"monitor-{self._id} informed server monitor is done")

    async def track(
        self, heartbeat_interval: Optional[float] = None
    ) -> AsyncGenerator[Event, None]:
        """Yield events from the internal event queue with optional heartbeats.

        Heartbeats are represented by None being yielded.

        Heartbeats stops being emitted after a CloseTrackerEvent is found."""
        _heartbeat_interval: Optional[float] = heartbeat_interval
        closetracker_received: bool = False
        while True:
            try:
                event = await asyncio.wait_for(
                    self._ee_to_monitor_queue.get(), timeout=_heartbeat_interval
                )

            except asyncio.TimeoutError:
                if closetracker_received:
                    logger.error("Evaluator did not send the TERMINATED event!")
                    self._ee_to_monitor_queue.task_done()
                    break
                continue
            if isinstance(event, EventSentinel):
                closetracker_received = True
                _heartbeat_interval = self._receiver_timeout
            else:
                # print(f"YIELDING EVENT {event=}")
                if type(event) is EETerminated:
                    logger.debug(f"monitor-{self._id} client received terminated")
                    self._ee_to_monitor_queue.task_done()
                    break
                yield event
            if event is not None:
                self._ee_to_monitor_queue.task_done()
