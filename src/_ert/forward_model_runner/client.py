import asyncio
import logging
import ssl
from typing import Any, AnyStr, Optional, Union

from typing_extensions import Self
from websockets.client import WebSocketClientProtocol, connect
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
        token: Optional[str] = None,
        cert: Optional[Union[str, bytes]] = None,
        max_retries: int = 10,
        timeout_multiplier: int = 5,
    ) -> None:
        if url is None:
            raise ValueError("url was None")
        self.url = url
        self.token = token
        self._extra_headers = Headers()
        if token is not None:
            self._extra_headers["token"] = token

        # Mimics the behavior of the ssl argument when connection to
        # websockets. If none is specified it will deduce based on the url,
        # if True it will enforce TLS, and if you want to use self signed
        # certificates you need to pass an ssl_context with the certificate
        # loaded.
        self._ssl_context: Optional[Union[bool, ssl.SSLContext]] = None
        if cert is not None:
            self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            self._ssl_context.load_verify_locations(cadata=cert)
        elif url.startswith("wss"):
            self._ssl_context = True

        self._max_retries = max_retries
        self._timeout_multiplier = timeout_multiplier
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.loop = new_event_loop()

    async def get_websocket(self) -> WebSocketClientProtocol:
        return await connect(
            self.url,
            ssl=self._ssl_context,
            extra_headers=self._extra_headers,
            open_timeout=60,
            ping_timeout=60,
            ping_interval=60,
            close_timeout=60,
        )

    async def _send(self, msg: AnyStr) -> None:
        for retry in range(self._max_retries + 1):
            try:
                if self.websocket is None:
                    self.websocket = await self.get_websocket()
                await self.websocket.send(msg)
                return
            except ConnectionClosedOK as exception:
                _error_msg = (
                    f"Connection closed received from the server {self.url}! "
                    f" Exception from {type(exception)}: {exception!s}"
                )
                raise ClientConnectionClosedOK(_error_msg) from exception
            except (
                InvalidHandshake,
                InvalidURI,
                OSError,
                asyncio.TimeoutError,
            ) as exception:
                if retry == self._max_retries:
                    _error_msg = (
                        f"Not able to establish the "
                        f"websocket connection {self.url}! Max retries reached!"
                        " Check for firewall issues."
                        f" Exception from {type(exception)}: {exception!s}"
                    )
                    raise ClientConnectionError(_error_msg) from exception
            except ConnectionClosedError as exception:
                if retry == self._max_retries:
                    _error_msg = (
                        f"Not been able to send the event"
                        f" to {self.url}! Max retries reached!"
                        f" Exception from {type(exception)}: {exception!s}"
                    )
                    raise ClientConnectionError(_error_msg) from exception
            await asyncio.sleep(0.2 + self._timeout_multiplier * retry)
            self.websocket = None

    def send(self, msg: AnyStr) -> None:
        self.loop.run_until_complete(self._send(msg))
