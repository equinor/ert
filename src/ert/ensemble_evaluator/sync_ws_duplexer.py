from __future__ import annotations

import asyncio
import contextlib
import queue
import ssl
import threading
import time
from concurrent.futures import CancelledError
from typing import AsyncIterable, Iterable, Iterator, Optional, Union

import websockets
from cloudevents.http import CloudEvent
from websockets.client import WebSocketClientProtocol
from websockets.datastructures import Headers
from websockets.typing import Data

from ert.async_utils import new_event_loop

from ._wait_for_evaluator import wait_for_evaluator


class SyncWebsocketDuplexer:
    """Class for communicating bi-directionally with a websocket using a
    synchronous API. Reentrant, but not thread-safe. One must call stop() after
    communication ends."""

    def __init__(
        self,
        uri: str,
        health_check_uri: str,
        cert: Union[str, bytes, None],
        token: Optional[str],
    ) -> None:
        self._uri = uri
        self._hc_uri = health_check_uri
        self._token = token
        self._extra_headers = Headers()
        if token is not None:
            self._extra_headers["token"] = token

        self.send_queue = queue.Queue()
        self.recv_queue = queue.Queue()

        # Mimics the behavior of the ssl argument when connection to
        # websockets. If none is specified it will deduce based on the url,
        # if True it will enforce TLS, and if you want to use self signed
        # certificates you need to pass an ssl_context with the certificate
        # loaded.
        self._cert = cert
        ssl_context: Optional[Union[bool, ssl.SSLContext]] = None
        if cert is not None:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.load_verify_locations(cadata=cert)
        else:
            ssl_context = True if self._uri.startswith("wss") else None
        self._ssl_context: Optional[Union[bool, ssl.SSLContext]] = ssl_context

        self._loop = new_event_loop()
        # self._connection: asyncio.Task[None] = self._loop.create_task(self._connect())
        self._connection: asyncio.Task[None] = self._loop.create_task(
            self._handle_client()
        )
        self._ws: Optional[WebSocketClientProtocol] = None
        self._loop_thread = threading.Thread(target=self._loop.run_forever)
        self._loop_thread.start()

    async def _handle_client(self):
        await wait_for_evaluator(
            base_url=self._hc_uri, token=self._token, cert=self._cert, timeout=5
        )
        async with websockets.client.connect(
            self._uri,
            ssl=self._ssl_context,
            extra_headers=self._extra_headers,
            max_size=2**26,
            max_queue=500,
            open_timeout=5,
            ping_timeout=60,
            ping_interval=60,
            close_timeout=60,
        ) as connect:
            while True:
                if not self.send_queue.empty():
                    message = await self._loop.run_in_executor(
                        None, self.send_queue.get
                    )
                    await connect.send(message)
                try:
                    received_message = await asyncio.wait_for(
                        connect.recv(), timeout=1.0
                    )
                    await self._loop.run_in_executor(
                        None, self.recv_queue.put, received_message
                    )
                except asyncio.TimeoutError:
                    pass

    def send(
        self,
        msg: Union[
            Data,
            Iterable[Data],
            AsyncIterable[Data],
        ],
    ) -> None:
        self.send_queue.put(msg)

    def receive(self) -> Iterator[CloudEvent]:
        """Create a generator with which you can iterate over incoming
        websocket messages."""
        while True:
            yield self.recv_queue.get()
