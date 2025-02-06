import asyncio
import logging
from contextlib import suppress

import pytest
import zmq
import zmq.asyncio

from _ert.events import EEUserCancel, EEUserDone, event_from_json
from _ert.forward_model_runner.client import (
    ACK_MSG,
    CONNECT_MSG,
    DISCONNECT_MSG,
    ClientConnectionError,
)
from ert.ensemble_evaluator import Monitor


def localhost_uri(port: int):
    return f"tcp://127.0.0.1:{port}"


async def async_zmq_server(port, handler, secret_key: bytes | None = None):
    zmq_context = zmq.asyncio.Context()
    router_socket = zmq_context.socket(zmq.ROUTER)
    if secret_key is not None:
        router_socket.curve_secretkey = secret_key
        router_socket.curve_publickey = zmq.curve_public(secret_key)
        router_socket.curve_server = True
    router_socket.setsockopt(zmq.LINGER, 0)
    router_socket.bind(f"tcp://*:{port}")
    await handler(router_socket)
    router_socket.close()
    zmq_context.destroy()


async def test_monitor_connects_and_disconnects_successfully(unused_tcp_port):
    monitor = Monitor(localhost_uri(unused_tcp_port))

    messages = []

    async def mock_event_handler(router_socket):
        nonlocal messages
        while True:
            dealer, _, frame = await router_socket.recv_multipart()
            await router_socket.send_multipart([dealer, b"", ACK_MSG])
            messages.append((dealer.decode("utf-8"), frame))
            if frame == DISCONNECT_MSG:
                break

    websocket_server_task = asyncio.create_task(
        async_zmq_server(unused_tcp_port, mock_event_handler)
    )
    async with monitor:
        pass
    await websocket_server_task
    dealer, msg = messages[0]
    assert dealer.startswith("client-")
    assert msg == CONNECT_MSG
    dealer, msg = messages[1]
    assert dealer.startswith("client-")
    assert msg == DISCONNECT_MSG


async def test_no_connection_established(monkeypatch, make_ee_config):
    ee_config = make_ee_config()
    monkeypatch.setattr(Monitor, "DEFAULT_MAX_RETRIES", 0)
    monitor = Monitor(ee_config.get_uri())
    monitor._ack_timeout = 0.1
    with pytest.raises(ClientConnectionError):
        async with monitor:
            pass


async def test_immediate_stop(unused_tcp_port):
    connected = False

    async def mock_event_handler(router_socket):
        nonlocal connected
        while True:
            dealer, _, frame = await router_socket.recv_multipart()
            await router_socket.send_multipart([dealer, b"", ACK_MSG])
            dealer = dealer.decode("utf-8")
            if frame == CONNECT_MSG:
                connected = True
            elif frame == DISCONNECT_MSG:
                connected = False
                return
            else:
                event = event_from_json(frame.decode("utf-8"))
                assert connected
                assert type(event) is EEUserDone

    websocket_server_task = asyncio.create_task(
        async_zmq_server(unused_tcp_port, mock_event_handler)
    )
    async with Monitor(localhost_uri(unused_tcp_port)) as monitor:
        assert connected is True
        await monitor.signal_done()
    await websocket_server_task
    assert connected is False


@pytest.mark.integration_test
async def test_unexpected_close_after_connection_successful(
    monkeypatch, unused_tcp_port
):
    monkeypatch.setattr(Monitor, "DEFAULT_MAX_RETRIES", 0)
    monkeypatch.setattr(Monitor, "DEFAULT_ACK_TIMEOUT", 0.5)

    async def mock_event_handler(router_socket):
        dealer, _, frame = await router_socket.recv_multipart()
        await router_socket.send_multipart([dealer, b"", ACK_MSG])
        dealer = dealer.decode("utf-8")
        assert dealer.startswith("client-")
        assert frame == CONNECT_MSG
        router_socket.close()

    websocket_server_task = asyncio.create_task(
        async_zmq_server(unused_tcp_port, mock_event_handler)
    )
    async with Monitor(localhost_uri(unused_tcp_port)) as monitor:
        with pytest.raises(ClientConnectionError):
            await monitor.signal_done()

    await websocket_server_task


@pytest.mark.parametrize(
    "correct_server_key",
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
async def test_that_monitor_cannot_connect_with_wrong_server_key(
    correct_server_key, monkeypatch, unused_tcp_port
):
    public_key, secret_key = zmq.curve_keypair()
    uri = localhost_uri(unused_tcp_port)
    token = public_key.decode("utf-8") if correct_server_key else None

    monkeypatch.setattr(Monitor, "DEFAULT_MAX_RETRIES", 0)
    monkeypatch.setattr(Monitor, "DEFAULT_ACK_TIMEOUT", 0.5)

    connected = False

    async def mock_event_handler(router_socket):
        nonlocal connected
        while True:
            dealer, _, frame = await router_socket.recv_multipart()
            await router_socket.send_multipart([dealer, b"", ACK_MSG])
            if frame == CONNECT_MSG:
                connected = True
            elif frame == DISCONNECT_MSG:
                connected = False
                return

    websocket_server_task = asyncio.create_task(
        async_zmq_server(unused_tcp_port, mock_event_handler, secret_key=secret_key)
    )
    if correct_server_key:
        async with Monitor(uri, token):
            assert connected
        assert connected is False
    else:
        with pytest.raises(ClientConnectionError):
            async with Monitor(uri, token):
                pass
        assert connected is False
        websocket_server_task.cancel()
    with suppress(asyncio.CancelledError):
        await websocket_server_task


async def test_that_monitor_track_can_exit_without_terminated_event_from_evaluator(
    unused_tcp_port, caplog
):
    caplog.set_level(logging.ERROR)
    uri = localhost_uri(unused_tcp_port)

    connected = False

    async def mock_event_handler(router_socket):
        nonlocal connected
        while True:
            dealer, _, frame = await router_socket.recv_multipart()
            await router_socket.send_multipart([dealer, b"", ACK_MSG])
            if frame == CONNECT_MSG:
                connected = True
            elif frame == DISCONNECT_MSG:
                connected = False
                return
            else:
                event = event_from_json(frame.decode("utf-8"))
                assert connected
                assert type(event) is EEUserCancel

    websocket_server_task = asyncio.create_task(
        async_zmq_server(unused_tcp_port, mock_event_handler)
    )

    async with Monitor(uri) as monitor:
        monitor._receiver_timeout = 0.1
        await monitor.signal_cancel()

        async for event in monitor.track():
            raise RuntimeError(f"Got unexpected event {event} after cancellation")

        assert ("Evaluator did not send the TERMINATED event!") in caplog.messages, (
            "Monitor receiver did not stop!"
        )

    await websocket_server_task


async def test_that_monitor_can_emit_heartbeats(unused_tcp_port):
    """BaseRunModel.run_monitor() depends on heartbeats to be able to
    exit anytime. A heartbeat is a None event.

    If the heartbeat is never sent, this test function will hang and then timeout."""
    uri = localhost_uri(unused_tcp_port)

    async def mock_event_handler(router_socket):
        while True:
            try:
                dealer, _, __ = await router_socket.recv_multipart()
                await router_socket.send_multipart([dealer, b"", ACK_MSG])
            except asyncio.CancelledError:
                break

    websocket_server_task = asyncio.create_task(
        async_zmq_server(unused_tcp_port, mock_event_handler)
    )

    async with Monitor(uri) as monitor:
        async for event in monitor.track(heartbeat_interval=0.001):
            if event is None:
                break

    if not websocket_server_task.done():
        websocket_server_task.cancel()
        asyncio.gather(websocket_server_task, return_exceptions=True)
