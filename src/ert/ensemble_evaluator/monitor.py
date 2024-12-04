from __future__ import annotations

import asyncio
import logging
import ssl
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Final, Optional, Union

import zmq.asyncio

from _ert.events import (
    EETerminated,
    EEUserCancel,
    EEUserDone,
    Event,
    event_from_json,
    event_to_json,
)

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
        self._receiver_task: Optional[asyncio.Task[None]] = None
        self._connected: asyncio.Future[None] = asyncio.Future()
        self._connection_timeout: float = 120.0
        self._receiver_timeout: float = 60.0
        # zmq connection
        self._zmq_context = zmq.asyncio.Context()
        self._socket = self._zmq_context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt_string(zmq.IDENTITY, f"client-{self._id}")
        print(f"{self._id=} wiith {ee_con_info.token=}")
        if ee_con_info.token is not None:
            client_public, client_secret = zmq.curve_keypair()
            self._socket.curve_secretkey = client_secret
            self._socket.curve_publickey = client_public
            self._socket.curve_serverkey = ee_con_info.token.encode("utf-8")

    async def __aenter__(self) -> "Monitor":
        try:
            await self.reconnect()
        except asyncio.TimeoutError as exc:
            await self._term()
            msg = "Couldn't establish connection with the ensemble evaluator!"
            logger.error(msg)
            raise RuntimeError(msg) from exc
        self._receiver_task = asyncio.create_task(self._receiver())
        return self

    async def _term(self) -> None:
        if self._receiver_task:
            await self._socket.send_multipart([b"", b"DISCONNECT"])
            self._socket.disconnect(self._ee_con_info.router_uri)
            if not self._receiver_task.done():
                self._receiver_task.cancel()
            await asyncio.gather(
                self._receiver_task,
                return_exceptions=True,
            )
        self._socket.close()
        self._zmq_context.term()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        await self._term()

    async def signal_cancel(self) -> None:
        await self._event_queue.put(Monitor._sentinel)
        logger.debug(f"monitor-{self._id} asking server to cancel...")

        cancel_event = EEUserCancel(monitor=self._id)
        await self._socket.send_multipart(
            [b"", event_to_json(cancel_event).encode("utf-8")]
        )
        logger.debug(f"monitor-{self._id} asked server to cancel")

    async def signal_done(self) -> None:
        await self._event_queue.put(Monitor._sentinel)
        logger.debug(f"monitor-{self._id} informing server monitor is done...")

        done_event = EEUserDone(monitor=self._id)
        await self._socket.send_multipart(
            [b"", event_to_json(done_event).encode("utf-8")]
        )
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

    async def reconnect(self) -> None:
        self._socket.connect(self._ee_con_info.router_uri)
        await self._socket.send_multipart([b"", b"CONNECT"])
        try:
            _, ack = await asyncio.wait_for(
                self._socket.recv_multipart(), timeout=self._connection_timeout
            )
            if ack.decode() != "ACK":
                raise asyncio.TimeoutError("No Ack for connect")
            print(f"{self._id=} MONITOR CONNECTED")
        except asyncio.TimeoutError:
            print("NO CONNECTION")
            logger.warning(
                f"Failed to get acknowledgment on the monitor {self._id} connect!"
            )
            raise

    async def _receiver(self) -> None:
        tls: Optional[ssl.SSLContext] = None
        if self._ee_con_info.cert:
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            tls.load_verify_locations(cadata=self._ee_con_info.cert)
        while True:
            try:
                _, raw_msg = await self._socket.recv_multipart()
                event = event_from_json(raw_msg.decode("utf-8"))
                await self._event_queue.put(event)
            except zmq.ZMQError as exc:
                # Handle disconnection or other ZMQ errors (reconnect or log)
                logger.debug(
                    f"ZeroMQ connection to EnsembleEvaluator went down, reconnecting: {exc}"
                )
                await self.reconnect()
