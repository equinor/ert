import asyncio
from contextlib import ExitStack
from http import HTTPStatus
from threading import Event, Thread
from unittest.mock import patch

import pytest
import websockets
from websockets.exceptions import ConnectionClosedOK

from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.ensemble_evaluator.sync_ws_duplexer import SyncWebsocketDuplexer


@pytest.fixture
def ws(event_loop: asyncio.AbstractEventLoop):
    t = Thread(target=event_loop.run_forever)
    t.start()

    async def _process_request(path, request_headers):
        if path == "/healthcheck":
            return HTTPStatus.OK, {}, b""

    def _start_ws(host: str, port: int, handler, ssl=None, sock=None):
        kwargs = {
            "process_request": _process_request,
            "ssl": ssl,
        }
        if sock:
            kwargs["sock"] = sock
        else:
            kwargs["host"] = host
            kwargs["port"] = port
        event_loop.call_soon_threadsafe(
            asyncio.ensure_future,
            websockets.server.serve(
                handler,
                **kwargs,
                open_timeout=10,
            ),
        )

    yield _start_ws
    event_loop.call_soon_threadsafe(event_loop.stop)
    t.join()


def test_immediate_stop(unused_tcp_port, ws):
    # if the duplexer is immediately stopped, it should be well behaved and close
    # the connection normally
    closed = Event()

    async def handler(websocket, path):
        try:
            await websocket.recv()
        except ConnectionClosedOK:
            closed.set()

    ws("localhost", unused_tcp_port, handler)

    duplexer = SyncWebsocketDuplexer(
        f"ws://localhost:{unused_tcp_port}",
        f"ws://localhost:{unused_tcp_port}",
        None,
        None,
    )

    duplexer.stop()

    assert closed.wait(10)


def test_failed_connection():
    with patch("ert.ensemble_evaluator.sync_ws_duplexer.wait_for_evaluator") as w:
        w.side_effect = OSError("expected OSError")
        with pytest.raises(OSError, match="expected OSError"):
            SyncWebsocketDuplexer("ws://localhost:0", "http://localhost:0", None, None)


def test_failed_send(unused_tcp_port, ws):
    async def handler(websocket, path):
        await websocket.recv()

    ws("localhost", unused_tcp_port, handler)
    duplexer = SyncWebsocketDuplexer(
        f"ws://localhost:{unused_tcp_port}",
        f"ws://localhost:{unused_tcp_port}",
        None,
        None,
    )
    with patch("websockets.WebSocketCommonProtocol.send") as send:
        send.side_effect = OSError("expected OSError")
        with pytest.raises(OSError, match="expected OSError"):
            duplexer.send("hello")


def test_unexpected_close(unused_tcp_port, ws):
    async def handler(websocket, path):
        await websocket.close()

    ws("localhost", unused_tcp_port, handler)
    with ExitStack() as stack:
        duplexer = SyncWebsocketDuplexer(
            f"ws://localhost:{unused_tcp_port}",
            f"ws://localhost:{unused_tcp_port}",
            None,
            None,
        )
        stack.callback(duplexer.stop)
        with pytest.raises(ConnectionClosedOK):
            next(duplexer.receive())


def test_receive(unused_tcp_port, ws):
    async def handler(websocket, path):
        await websocket.send("Hello World")

    ws("localhost", unused_tcp_port, handler)
    with ExitStack() as stack:
        duplexer = SyncWebsocketDuplexer(
            f"ws://localhost:{unused_tcp_port}",
            f"ws://localhost:{unused_tcp_port}",
            None,
            None,
        )
        stack.callback(duplexer.stop)
        assert next(duplexer.receive()) == "Hello World"


def test_echo(unused_tcp_port, ws):
    async def handler(websocket, path):
        msg = await websocket.recv()
        await websocket.send(msg)

    ws("localhost", unused_tcp_port, handler)
    with ExitStack() as stack:
        duplexer = SyncWebsocketDuplexer(
            f"ws://localhost:{unused_tcp_port}",
            f"ws://localhost:{unused_tcp_port}",
            None,
            None,
        )
        stack.callback(duplexer.stop)

        duplexer.send("Hello World")
        assert next(duplexer.receive()) == "Hello World"


def test_generator(unused_tcp_port, ws):
    async def handler(websocket, path):
        await websocket.send("one")
        await websocket.send("two")
        await websocket.send("three")
        await websocket.send("four")

    ws("localhost", unused_tcp_port, handler)
    with ExitStack() as stack:
        duplexer = SyncWebsocketDuplexer(
            f"ws://localhost:{unused_tcp_port}",
            f"ws://localhost:{unused_tcp_port}",
            None,
            None,
        )
        stack.callback(duplexer.stop)

        expected = ["one", "two", "three"]
        for msg in duplexer.receive():
            assert msg == expected.pop(0)

            # Cause a GeneratorExit
            if len(expected) == 1:
                break


def test_secure_echo(ws):
    custom_port_range = range(1024, 65535)
    config = EvaluatorServerConfig(
        custom_port_range=custom_port_range, custom_host="127.0.0.1"
    )

    async def handler(websocket, path):
        msg = await websocket.recv()
        await websocket.send(msg)

    ws(
        config.host,
        config.port,
        handler,
        ssl=config.get_server_ssl_context(),
        sock=config.get_socket(),
    )
    with ExitStack() as stack:
        duplexer = SyncWebsocketDuplexer(
            f"wss://{config.host}:{config.port}",
            f"wss://{config.host}:{config.port}",
            cert=config.cert,
            token=None,
        )
        stack.callback(duplexer.stop)
        duplexer.send("Hello Secure World")
        assert next(duplexer.receive()) == "Hello Secure World"
