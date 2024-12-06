from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional, Union

import zmq
import zmq.asyncio
from typing_extensions import Self

from _ert.async_utils import new_event_loop

logger = logging.getLogger(__name__)


class ClientConnectionError(Exception):
    pass


class ClientConnectionClosedOK(Exception):
    pass


class Client:
    DEFAULT_MAX_RETRIES = 10
    DEFAULT_TIMEOUT_MULTIPLIER = 5
    CONNECTION_TIMEOUT = 60

    def __enter__(self) -> Self:
        self.loop.run_until_complete(self.__aenter__())
        return self

    def term(self) -> None:
        self.socket.close()
        self.context.term()

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self.loop.run_until_complete(self.__aexit__(exc_type, exc_value, exc_traceback))
        self.loop.close()

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self, exc_type: Any, exc_value: Any, exc_traceback: Any
    ) -> None:
        await self._send("DISCONNECT")
        self.socket.disconnect(self.url)
        await self._term_receiver_task()
        self.term()

    async def _term_receiver_task(self) -> None:
        if self._receiver_task and not self._receiver_task.done():
            self._receiver_task.cancel()
            await asyncio.gather(self._receiver_task, return_exceptions=True)
        self._receiver_task = None

    def __init__(
        self,
        url: str,
        token: Optional[str] = None,
        cert: Optional[Union[str, bytes]] = None,
        connection_timeout: float = 5.0,
        dealer_name: Optional[str] = None,
    ) -> None:
        self._connection_timeout = connection_timeout
        self.url = url
        self.token = token

        # Set up ZeroMQ context and socke
        self._ack_event: asyncio.Event = asyncio.Event()
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        if dealer_name is None:
            self.dealer_id = f"dispatch-{uuid.uuid4().hex[:8]}"
        else:
            self.dealer_id = dealer_name
        self.socket.setsockopt_string(zmq.IDENTITY, self.dealer_id)
        print(f"Created: {self.dealer_id=} {token=} {self._connection_timeout=}")
        if token is not None:
            client_public, client_secret = zmq.curve_keypair()
            self.socket.curve_secretkey = client_secret
            self.socket.curve_publickey = client_public
            self.socket.curve_serverkey = token.encode("utf-8")

        self.loop = new_event_loop()
        self._receiver_task: Optional[asyncio.Task[None]] = None

    async def connect(self) -> None:
        self.socket.connect(self.url)
        await self._term_receiver_task()
        self._receiver_task = asyncio.create_task(self._receiver())
        try:
            await self._send("CONNECT", max_retries=1)
        except ClientConnectionError:
            await self._term_receiver_task()
            self.term()
            raise

    def send(
        self, messages: str | list[str], max_retries: int = DEFAULT_MAX_RETRIES
    ) -> None:
        self.loop.run_until_complete(self._send(messages, max_retries))

    async def process_message(self, msg: str) -> None:
        pass

    async def _receiver(self) -> None:
        while True:
            try:
                _, raw_msg = await self.socket.recv_multipart()
                if raw_msg == b"ACK":
                    self._ack_event.set()
                else:
                    await self.process_message(raw_msg.decode("utf-8"))
            except zmq.ZMQError as exc:
                logger.debug(
                    f"{self.dealer_id} connection to evaluator went down, reconnecting: {exc}"
                )
                await asyncio.sleep(1)
                self.socket.connect(self.url)

    async def _send(
        self, messages: str | list[str], max_retries: int = DEFAULT_MAX_RETRIES
    ) -> None:
        self._ack_event.clear()
        if isinstance(messages, str):
            messages = [messages]

        backoff = 1

        while max_retries > 0:
            try:
                await self.socket.send_multipart(
                    [b""] + [message.encode("utf-8") for message in messages]
                )
                try:
                    await asyncio.wait_for(
                        self._ack_event.wait(), timeout=self._connection_timeout
                    )
                    return
                except asyncio.TimeoutError:
                    logger.warning(
                        f"{self.dealer_id} failed to get acknowledgment on the {messages}. Resending."
                    )
            except zmq.ZMQError as exc:
                logger.debug(
                    f"{self.dealer_id} connection to evaluator went down, reconnecting: {exc}"
                )
                await asyncio.sleep(1)
                self.socket.connect(self.url)
            except asyncio.CancelledError:
                self.term()
                raise

            max_retries -= 1
            if max_retries > 0:
                logger.info(f"Retrying... ({max_retries} attempts left)")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10)  # Exponential backoff

        raise ClientConnectionError(
            f"{self.dealer_id} Failed to send {messages=} after retries."
        )
