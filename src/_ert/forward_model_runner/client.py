import asyncio
import logging
import ssl
from typing import Any, AnyStr, Self

from websockets.asyncio.client import ClientConnection, connect
from websockets.datastructures import Headers
from websockets.exceptions import (
    ConnectionClosedError,
    ConnectionClosedOK,
    InvalidHandshake,
    InvalidURI,
)

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
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        if self.websocket is not None:
            self.loop.run_until_complete(self.websocket.close())
        self.loop.close()

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(
        self, exc_type: Any, exc_value: Any, exc_traceback: Any
    ) -> None:
        if self.websocket is not None:
            await self.websocket.close()

    def __init__(
        self,
        url: str,
        token: str | None = None,
        cert: str | bytes | None = None,
        max_retries: int | None = None,
        timeout_multiplier: int | None = None,
    ) -> None:
        if max_retries is None:
            max_retries = self.DEFAULT_MAX_RETRIES
        if timeout_multiplier is None:
            timeout_multiplier = self.DEFAULT_TIMEOUT_MULTIPLIER
        if url is None:
            raise ValueError("url was None")
        self.url = url
        self.token = token
        self._additional_headers = Headers()
        if token is not None:
            self._additional_headers["token"] = token

        # Mimics the behavior of the ssl argument when connection to
        # websockets. If none is specified it will deduce based on the url,
        # if True it will enforce TLS, and if you want to use self signed
        # certificates you need to pass an ssl_context with the certificate
        # loaded.
        self._ssl_context: bool | ssl.SSLContext | None = None
        if cert is not None:
            self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            self._ssl_context.load_verify_locations(cadata=cert)
        elif url.startswith("wss"):
            self._ssl_context = True

        self._max_retries = max_retries
        self._timeout_multiplier = timeout_multiplier
        self.websocket: ClientConnection | None = None
        self.loop = new_event_loop()

    async def get_websocket(self) -> ClientConnection:
        return await connect(
            self.url,
            ssl=self._ssl_context,
            additional_headers=self._additional_headers,
            open_timeout=self.CONNECTION_TIMEOUT,
            ping_timeout=self.CONNECTION_TIMEOUT,
            ping_interval=self.CONNECTION_TIMEOUT,
            close_timeout=self.CONNECTION_TIMEOUT,
        )

    async def _send(self, msg: AnyStr) -> None:
        for retry in range(self._max_retries + 1):
            try:
                if self.websocket is None:
                    self.websocket = await self.get_websocket()
                await self.websocket.send(msg)
                return
            except ConnectionClosedOK as exception:
                error_msg = (
                    f"Connection closed received from the server {self.url}! "
                    f" Exception from {type(exception)}: {exception!s}"
                )
                raise ClientConnectionClosedOK(error_msg) from exception
            except (TimeoutError, InvalidHandshake, InvalidURI, OSError) as exception:
                if retry == self._max_retries:
                    error_msg = (
                        f"Not able to establish the "
                        f"websocket connection {self.url}! Max retries reached!"
                        " Check for firewall issues."
                        f" Exception from {type(exception)}: {exception!s}"
                    )
                    raise ClientConnectionError(error_msg) from exception
            except ConnectionClosedError as exception:
                if retry == self._max_retries:
                    error_msg = (
                        f"Not been able to send the event"
                        f" to {self.url}! Max retries reached!"
                        f" Exception from {type(exception)}: {exception!s}"
                    )
                    raise ClientConnectionError(error_msg) from exception
            await asyncio.sleep(0.2 + self._timeout_multiplier * retry)
            self.websocket = None

    def send(self, msg: AnyStr) -> None:
        self.loop.run_until_complete(self._send(msg))
