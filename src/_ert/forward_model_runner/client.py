from __future__ import annotations

import asyncio
import logging
import types
import uuid
from typing import Self

import zmq
import zmq.asyncio

logger = logging.getLogger(__name__)


class ClientConnectionError(Exception):
    pass


CONNECT_MSG = b"CONNECT"
DISCONNECT_MSG = b"DISCONNECT"
ACK_MSG = b"ACK"
TERMINATE_MSG = b"TERMINATE"
HEARTBEAT_INTERVAL_MS = 5_000
HEARTBEAT_TIMEOUT_MS = 30_000


def enable_heartbeating(socket: zmq.asyncio.Socket) -> None:
    socket.setsockopt(zmq.HEARTBEAT_IVL, HEARTBEAT_INTERVAL_MS)
    socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, HEARTBEAT_TIMEOUT_MS)
    socket.setsockopt(zmq.HEARTBEAT_TTL, HEARTBEAT_TIMEOUT_MS)


class Client:
    DEFAULT_MAX_RETRIES = 10
    DEFAULT_ACK_TIMEOUT = 5

    def __init__(
        self,
        url: str,
        token: str | None = None,
        dealer_name: str | None = None,
        ack_timeout: float | None = None,
    ) -> None:
        self._ack_timeout = ack_timeout or self.DEFAULT_ACK_TIMEOUT
        self.url = url
        self.token = token

        self._ack_event: asyncio.Event = asyncio.Event()
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.DEALER)
        # this is to avoid blocking the event loop when closing the socket
        # wherein the linger is set to 0 to discard all messages in the queue
        self.socket.setsockopt(zmq.LINGER, 0)
        enable_heartbeating(self.socket)
        self.dealer_id = dealer_name or f"dispatch-{uuid.uuid4().hex[:8]}"
        self.received_terminate_message: asyncio.Event = asyncio.Event()
        self.socket.setsockopt_string(zmq.IDENTITY, self.dealer_id)

        if token is not None:
            client_public, client_secret = zmq.curve_keypair()
            self.socket.curve_secretkey = client_secret
            self.socket.curve_publickey = client_public
            self.socket.curve_serverkey = token.encode("utf-8")

        self._receiver_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        try:
            await self.connect()
        except ClientConnectionError:
            logger.error(
                "No ack for dealer connection. Connection was not established!"
            )
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: types.TracebackType | None,
    ) -> None:
        try:
            await self.send(DISCONNECT_MSG)
        except ClientConnectionError:
            logger.error("No ack for dealer disconnection. Connection is down!")
        finally:
            self.socket.disconnect(self.url)
            await self._term_receiver_task()
            self.term()

    def term(self) -> None:
        self.socket.close()
        self.context.term()

    async def _term_receiver_task(self) -> None:
        if self._receiver_task and not self._receiver_task.done():
            self._receiver_task.cancel()
            await asyncio.gather(self._receiver_task, return_exceptions=True)
            self._receiver_task = None

    async def connect(self) -> None:
        self.socket.connect(self.url)
        await self._term_receiver_task()
        self._receiver_task = asyncio.create_task(self._receiver())
        try:
            await self.send(CONNECT_MSG)
        except ClientConnectionError:
            await self._term_receiver_task()
            self.term()
            raise

    async def process_message(self, msg: str) -> None:
        raise NotImplementedError("Only monitor can receive messages!")

    async def _receiver(self) -> None:
        while True:
            try:
                _, raw_msg = await self.socket.recv_multipart()
            except zmq.ZMQError as exc:
                logger.warning(
                    f"{self.dealer_id} stopped receiving from evaluator: {exc}"
                )
                return
            if raw_msg == ACK_MSG:
                self._ack_event.set()
            elif raw_msg == TERMINATE_MSG:
                self.received_terminate_message.set()
            else:
                await self.process_message(raw_msg.decode("utf-8"))

    async def send(self, message: str | bytes, retries: int | None = None) -> None:
        self._ack_event.clear()

        if isinstance(message, str):
            message = message.encode("utf-8")

        backoff = 1
        if retries is None:
            retries = self.DEFAULT_MAX_RETRIES
        while retries >= 0:
            try:  # ruff: ignore[too-many-statements-in-try-clause]
                await self.socket.send_multipart([b"", message])
                try:
                    await asyncio.wait_for(
                        self._ack_event.wait(), timeout=self._ack_timeout
                    )
                except TimeoutError:
                    logger.warning(
                        f"{self.dealer_id} failed to get acknowledgment on "
                        f"the {message!r}. Resending."
                    )
                else:
                    return
            except zmq.ZMQError as exc:
                logger.debug(
                    f"{self.dealer_id} connection to evaluator went down, "
                    f"reconnecting: {exc}"
                )
            except asyncio.CancelledError:
                raise

            if retries > 0:
                logger.info(f"Retrying... ({retries} attempts left)")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10)  # Exponential backoff
            retries -= 1
        raise ClientConnectionError(
            f"{self.dealer_id} Failed to send {message!r} to {self.url} after retrying!"
        )
