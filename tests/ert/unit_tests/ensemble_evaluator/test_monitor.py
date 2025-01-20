import asyncio
import logging

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
from ert.ensemble_evaluator.config import EvaluatorConnectionInfo


async def async_zmq_server(port, handler):
    zmq_context = zmq.asyncio.Context()
    router_socket = zmq_context.socket(zmq.ROUTER)
    router_socket.setsockopt(zmq.LINGER, 0)
    router_socket.bind(f"tcp://*:{port}")
    await handler(router_socket)
    router_socket.close()
    zmq_context.destroy()


async def test_monitor_connects_and_disconnects_successfully(unused_tcp_port):
    ee_con_info = EvaluatorConnectionInfo(f"tcp://127.0.0.1:{unused_tcp_port}")
    monitor = Monitor(ee_con_info)

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


async def test_no_connection_established(make_ee_config):
    ee_config = make_ee_config()
    monitor = Monitor(ee_config.get_connection_info())
    monitor._ack_timeout = 0.1
    with pytest.raises(ClientConnectionError):
        async with monitor:
            pass


async def test_immediate_stop(unused_tcp_port):
    ee_con_info = EvaluatorConnectionInfo(f"tcp://127.0.0.1:{unused_tcp_port}")

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
    async with Monitor(ee_con_info) as monitor:
        assert connected is True
        await monitor.signal_done()
    await websocket_server_task
    assert connected is False


@pytest.mark.integration_test
async def test_unexpected_close_after_connection_successful(
    monkeypatch, unused_tcp_port
):
    ee_con_info = EvaluatorConnectionInfo(f"tcp://127.0.0.1:{unused_tcp_port}")

    monkeypatch.setattr(Monitor, "DEFAULT_MAX_RETRIES", 0)
    monkeypatch.setattr(Monitor, "DEFAULT_ACK_TIMEOUT", 1)

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
    async with Monitor(ee_con_info) as monitor:
        with pytest.raises(ClientConnectionError):
            await monitor.signal_done()

    await websocket_server_task


async def test_that_monitor_track_can_exit_without_terminated_event_from_evaluator(
    unused_tcp_port, caplog
):
    caplog.set_level(logging.ERROR)
    ee_con_info = EvaluatorConnectionInfo(f"tcp://127.0.0.1:{unused_tcp_port}")

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

    async with Monitor(ee_con_info) as monitor:
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
    ee_con_info = EvaluatorConnectionInfo(f"tcp://127.0.0.1:{unused_tcp_port}")

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

    async with Monitor(ee_con_info) as monitor:
        async for event in monitor.track(heartbeat_interval=0.001):
            if event is None:
                break

    if not websocket_server_task.done():
        websocket_server_task.cancel()
        asyncio.gather(websocket_server_task, return_exceptions=True)
