import asyncio
import logging
import ssl
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Final, Optional, Union

from aiohttp import ClientError
from websockets import ConnectionClosed, Headers, WebSocketClientProtocol
from websockets.client import connect

from _ert.events import (
    EETerminated,
    EEUserCancel,
    EEUserDone,
    Event,
    event_from_json,
    event_to_json,
)
from ert.ensemble_evaluator._wait_for_evaluator import wait_for_evaluator

if TYPE_CHECKING:
    from ert.ensemble_evaluator.evaluator_connection_info import EvaluatorConnectionInfo


logger = logging.getLogger(__name__)


class EventSentinel:
    pass


class Monitor:
    _sentinel: Final = EventSentinel()

    def __init__(self, ee_con_info: "EvaluatorConnectionInfo") -> None:
        self._ee_con_info = ee_con_info
        self._id = str(uuid.uuid1()).split("-", maxsplit=1)[0]
        self._event_queue: asyncio.Queue[Union[Event, EventSentinel]] = asyncio.Queue()
        self._connection: Optional[WebSocketClientProtocol] = None
        self._receiver_task: Optional[asyncio.Task[None]] = None
        self._connected: asyncio.Event = asyncio.Event()
        self._connection_timeout: float = 120.0
        self._receiver_timeout: float = 60.0

    async def __aenter__(self) -> "Monitor":
        self._receiver_task = asyncio.create_task(self._receiver())
        try:
            await asyncio.wait_for(
                self._connected.wait(), timeout=self._connection_timeout
            )
        except asyncio.TimeoutError as exc:
            msg = "Couldn't establish connection with the ensemble evaluator!"
            logger.error(msg)
            self._receiver_task.cancel()
            raise RuntimeError(msg) from exc
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._receiver_task:
            if not self._receiver_task.done():
                self._receiver_task.cancel()
            # we are done and not interested in errors when cancelling
            await asyncio.gather(
                self._receiver_task,
                return_exceptions=True,
            )

        if self._connection:
            await self._connection.close()

    async def signal_cancel(self) -> None:
        if not self._connection:
            return
        await self._event_queue.put(Monitor._sentinel)
        logger.debug(f"monitor-{self._id} asking server to cancel...")

        cancel_event = EEUserCancel(monitor=self._id)
        await self._connection.send(event_to_json(cancel_event))
        logger.debug(f"monitor-{self._id} asked server to cancel")

    async def signal_done(self) -> None:
        if not self._connection:
            return
        await self._event_queue.put(Monitor._sentinel)
        logger.debug(f"monitor-{self._id} informing server monitor is done...")

        done_event = EEUserDone(monitor=self._id)
        await self._connection.send(event_to_json(done_event))
        logger.debug(f"monitor-{self._id} informed server monitor is done")

    async def track(
        self, heartbeat_interval: Optional[float] = None
    ) -> AsyncGenerator[Optional[Event], None]:
        """Yield events from the internal event queue with optional heartbeats.

        Heartbeats are represented by None being yielded.

        Heartbeats stops being emitted after a CloseTrackerEvent is found."""
        _heartbeat_interval: Optional[float] = heartbeat_interval
        closetracker_received: bool = False
        while True:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=_heartbeat_interval
                )
            except asyncio.TimeoutError:
                if closetracker_received:
                    logger.error("Evaluator did not send the TERMINATED event!")
                    break
                event = None
            if isinstance(event, EventSentinel):
                closetracker_received = True
                _heartbeat_interval = self._receiver_timeout
            else:
                yield event
                if type(event) is EETerminated:
                    logger.debug(f"monitor-{self._id} client received terminated")
                    break
            if event is not None:
                self._event_queue.task_done()

    async def _receiver(self) -> None:
        tls: Optional[ssl.SSLContext] = None
        if self._ee_con_info.cert:
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            tls.load_verify_locations(cadata=self._ee_con_info.cert)
        headers = Headers()
        if self._ee_con_info.token:
            headers["token"] = self._ee_con_info.token

        await wait_for_evaluator(
            base_url=self._ee_con_info.url,
            token=self._ee_con_info.token,
            cert=self._ee_con_info.cert,
            timeout=5,
        )
        async for conn in connect(
            self._ee_con_info.client_uri,
            ssl=tls,
            extra_headers=headers,
            max_size=2**26,
            max_queue=500,
            open_timeout=5,
            ping_timeout=60,
            ping_interval=60,
            close_timeout=60,
        ):
            try:
                self._connection = conn
                self._connected.set()
                async for raw_msg in self._connection:
                    event = event_from_json(raw_msg)
                    await self._event_queue.put(event)
            except (ConnectionRefusedError, ConnectionClosed, ClientError) as exc:
                self._connection = None
                self._connected.clear()
                logger.debug(
                    f"Monitor connection to EnsembleEvaluator went down, reconnecting: {exc}"
                )
