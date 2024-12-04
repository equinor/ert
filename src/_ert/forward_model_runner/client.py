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
        self.loop.run_until_complete(self.reconnect())
        return self

    def term(self) -> None:
        self.socket.close()
        self.context.term()
        self.loop.close()

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self.send("DISCONNECT")
        self.socket.disconnect(self.url)
        self.term()

    async def __aenter__(self) -> Self:
        await self.reconnect()
        return self

    async def __aexit__(
        self, exc_type: Any, exc_value: Any, exc_traceback: Any
    ) -> None:
        await self._send("DISCONNECT")
        self.socket.disconnect(self.url)
        self.term()

    def __init__(
        self,
        url: str,
        token: Optional[str] = None,
        cert: Optional[Union[str, bytes]] = None,
        max_retries: int = 10,
        connection_timeout: float = 5.0,
        dealer_name: Optional[str] = None,
    ) -> None:
        if max_retries is None:
            max_retries = self.DEFAULT_MAX_RETRIES
        self._connection_timeout = connection_timeout
        self.url = url
        self.token = token

        # Set up ZeroMQ context and socket
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        if dealer_name is None:
            dispatch_id = f"dispatch-{uuid.uuid4().hex[:8]}"
        else:
            dispatch_id = dealer_name
        self.dispatch_id = dispatch_id
        self.socket.setsockopt_string(zmq.IDENTITY, dispatch_id)
        print(f"{self.dispatch_id} {token}")
        if token is not None:
            client_public, client_secret = zmq.curve_keypair()
            self.socket.curve_secretkey = client_secret
            self.socket.curve_publickey = client_public
            self.socket.curve_serverkey = token.encode("utf-8")

        self._max_retries = max_retries
        self.loop = new_event_loop()

    async def reconnect(self) -> None:
        self.socket.connect(self.url)
        print(f"{self.dispatch_id=} CONNECTING to {self.url=}")
        try:
            await self._send("CONNECT", max_retries=1)
            _, ack = await asyncio.wait_for(
                self.socket.recv_multipart(), timeout=self._connection_timeout
            )
            if ack.decode() != "ACK":
                raise ClientConnectionError("No Ack for connect")
            print(f"{self.dispatch_id=} CONNECTED to {self.url=}")
        except asyncio.TimeoutError as exc:
            logger.warning("Failed to get acknowledgment on dealer connect!")
            self.term()
            raise ClientConnectionError(
                "Connection to evaluator not established!"
            ) from exc

    def send(
        self, messages: str | list[str], max_retries: Optional[int] = None
    ) -> None:
        self.loop.run_until_complete(self._send(messages, max_retries))

    async def _send(
        self, messages: str | list[str], max_retries: Optional[int] = None
    ) -> None:
        if isinstance(messages, str):
            messages = [messages]

        retries = max_retries or self._max_retries
        backoff = 1

        while retries > 0:
            try:
                await self.socket.send_multipart(
                    [b""] + [message.encode("utf-8") for message in messages]
                )

                # Wait for acknowledgment
                try:
                    _, ack = await asyncio.wait_for(
                        self.socket.recv_multipart(), timeout=self._connection_timeout
                    )
                    if ack.decode() == "ACK":
                        logger.info("Message acknowledged.")
                        print(f"message sent {messages=} from {self.dispatch_id=}")
                        return
                    logger.warning(
                        "Got acknowledgment but not the expected message. Resending."
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Failed to get acknowledgment on the message. Resending."
                    )

            except zmq.ZMQError as e:
                logger.warning(f"ZMQ error occurred: {e}. Reconnecting...")
                await self.reconnect()
            except asyncio.CancelledError:
                self.term()
                raise

            retries -= 1
            if retries > 0:
                logger.info(f"Retrying... ({retries} attempts left)")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10)  # Exponential backoff

        raise ClientConnectionError("Failed to send message after retries.")
