import asyncio
import logging
from http import HTTPStatus
from urllib.parse import urlparse

import pytest
from websockets import server
from websockets.exceptions import ConnectionClosedOK

from _ert.events import EEUserCancel, EEUserDone, event_from_json
from ert.ensemble_evaluator import Monitor
from ert.ensemble_evaluator.config import EvaluatorConnectionInfo


async def _mock_ws(
    set_when_done: asyncio.Event, handler, ee_config: EvaluatorConnectionInfo
):
    async def process_request(path, request_headers):
        if path == "/healthcheck":
            return HTTPStatus.OK, {}, b""

    url = urlparse(ee_config.url)
    async with server.serve(
        handler, url.hostname, url.port, process_request=process_request
    ):
        await set_when_done.wait()


async def test_no_connection_established(make_ee_config):
    ee_config = make_ee_config()
    monitor = Monitor(ee_config.get_connection_info())
    monitor._connection_timeout = 0.1
    with pytest.raises(
        RuntimeError, match="Couldn't establish connection with the ensemble evaluator!"
    ):
        async with monitor:
            pass


async def test_immediate_stop(unused_tcp_port):
    ee_con_info = EvaluatorConnectionInfo(f"ws://127.0.0.1:{unused_tcp_port}")

    set_when_done = asyncio.Event()

    async def mock_ws_event_handler(websocket):
        async for raw_msg in websocket:
            event = event_from_json(raw_msg)
            assert type(event) is EEUserDone
            break
        await websocket.close()

    websocket_server_task = asyncio.create_task(
        _mock_ws(set_when_done, mock_ws_event_handler, ee_con_info)
    )
    async with Monitor(ee_con_info) as monitor:
        await monitor.signal_done()
    set_when_done.set()
    await websocket_server_task


async def test_unexpected_close(unused_tcp_port):
    ee_con_info = EvaluatorConnectionInfo(f"ws://127.0.0.1:{unused_tcp_port}")

    set_when_done = asyncio.Event()
    socket_closed = asyncio.Event()

    async def mock_ws_event_handler(websocket):
        await websocket.close()
        socket_closed.set()

    websocket_server_task = asyncio.create_task(
        _mock_ws(set_when_done, mock_ws_event_handler, ee_con_info)
    )
    async with Monitor(ee_con_info) as monitor:
        # this expects Event send to fail
        # but no attempt on resubmitting
        # since connection closed via websocket.close
        with pytest.raises(ConnectionClosedOK):
            await socket_closed.wait()
            await monitor.signal_done()

    set_when_done.set()
    await websocket_server_task


async def test_that_monitor_track_can_exit_without_terminated_event_from_evaluator(
    unused_tcp_port, caplog
):
    caplog.set_level(logging.ERROR)
    ee_con_info = EvaluatorConnectionInfo(f"ws://127.0.0.1:{unused_tcp_port}")

    set_when_done = asyncio.Event()

    async def mock_ws_event_handler(websocket):
        async for raw_msg in websocket:
            event = event_from_json(raw_msg)
            assert type(event) is EEUserCancel
            break
        await websocket.close()

    websocket_server_task = asyncio.create_task(
        _mock_ws(set_when_done, mock_ws_event_handler, ee_con_info)
    )
    async with Monitor(ee_con_info) as monitor:
        monitor._receiver_timeout = 0.1
        await monitor.signal_cancel()

        async for event in monitor.track():
            raise RuntimeError(f"Got unexpected event {event} after cancellation")

        assert (
            "Evaluator did not send the TERMINATED event!"
        ) in caplog.messages, "Monitor receiver did not stop!"

    set_when_done.set()
    await websocket_server_task


async def test_that_monitor_can_emit_heartbeats(unused_tcp_port):
    """BaseRunModel.run_monitor() depends on heartbeats to be able to
    exit anytime. A heartbeat is a None event.

    If the heartbeat is never sent, this test function will hang and then timeout."""
    ee_con_info = EvaluatorConnectionInfo(f"ws://127.0.0.1:{unused_tcp_port}")

    set_when_done = asyncio.Event()
    websocket_server_task = asyncio.create_task(
        _mock_ws(set_when_done, None, ee_con_info)
    )

    async with Monitor(ee_con_info) as monitor:
        async for event in monitor.track(heartbeat_interval=0.001):
            if event is None:
                break

    set_when_done.set()  # shuts down websocket server
    await websocket_server_task
