import asyncio
import logging

import pytest
import zmq
import zmq.asyncio

from _ert.events import EEUserCancel, EEUserDone, event_from_json
from ert.ensemble_evaluator import Monitor
from ert.ensemble_evaluator.config import EvaluatorConnectionInfo


async def async_zmq_server(port, handler):
    zmq_context = zmq.asyncio.Context()  # type: ignore
    router_socket = zmq_context.socket(zmq.ROUTER)
    router_socket.setsockopt(zmq.LINGER, 0)
    router_socket.bind(f"tcp://*:{port}")
    await handler(router_socket)
    router_socket.close()
    zmq_context.destroy()


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
    ee_con_info = EvaluatorConnectionInfo(f"tcp://127.0.0.1:{unused_tcp_port}")

    connected = False

    async def mock_event_handler(router_socket):
        nonlocal connected
        while True:
            dealer, _, *frames = await router_socket.recv_multipart()
            dealer = dealer.decode("utf-8")
            for frame in frames:
                frame = frame.decode("utf-8")
                assert dealer.startswith("client-")
                if frame == "CONNECT":
                    await router_socket.send_multipart(
                        [dealer.encode("utf-8"), b"", b"ACK"]
                    )
                    connected = True
                elif frame == "DISCONNECT":
                    connected = False
                    print(connected)
                    return
                else:
                    event = event_from_json(frame)
                    print(f"{event=}")
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


# TODO: refactor
# async def test_unexpected_close(unused_tcp_port):
#     ee_con_info = EvaluatorConnectionInfo(f"tcp://127.0.0.1:{unused_tcp_port}")

#     async def mock_event_handler(router_socket):
#         router_socket.close()

#     websocket_server_task = asyncio.create_task(
#         async_zmq_server(unused_tcp_port, mock_event_handler)
#     )
#     async with Monitor(ee_con_info) as monitor:
#         await monitor.signal_done()

#     await websocket_server_task


async def test_that_monitor_track_can_exit_without_terminated_event_from_evaluator(
    unused_tcp_port, caplog
):
    caplog.set_level(logging.ERROR)
    ee_con_info = EvaluatorConnectionInfo(f"tcp://127.0.0.1:{unused_tcp_port}")

    connected = False

    async def mock_event_handler(router_socket):
        nonlocal connected
        while True:
            dealer, _, *frames = await router_socket.recv_multipart()
            dealer = dealer.decode("utf-8")
            for frame in frames:
                frame = frame.decode("utf-8")
                assert dealer.startswith("client-")
                if frame == "CONNECT":
                    await router_socket.send_multipart(
                        [dealer.encode("utf-8"), b"", b"ACK"]
                    )
                    connected = True
                elif frame == "DISCONNECT":
                    connected = False
                    return
                else:
                    event = event_from_json(frame)
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

        assert (
            "Evaluator did not send the TERMINATED event!"
        ) in caplog.messages, "Monitor receiver did not stop!"

    await websocket_server_task


async def test_that_monitor_can_emit_heartbeats(unused_tcp_port):
    """BaseRunModel.run_monitor() depends on heartbeats to be able to
    exit anytime. A heartbeat is a None event.

    If the heartbeat is never sent, this test function will hang and then timeout."""
    ee_con_info = EvaluatorConnectionInfo(f"tcp://127.0.0.1:{unused_tcp_port}")

    set_when_done = asyncio.Event()

    async def mock_event_handler(router_socket):
        dealer, _, *frames = await router_socket.recv_multipart()
        dealer = dealer.decode("utf-8")
        for frame in frames:
            frame = frame.decode("utf-8")
            if frame == "CONNECT":
                await router_socket.send_multipart(
                    [dealer.encode("utf-8"), b"", b"ACK"]
                )
        await set_when_done.wait()

    websocket_server_task = asyncio.create_task(
        async_zmq_server(unused_tcp_port, mock_event_handler)
    )

    async with Monitor(ee_con_info) as monitor:
        async for event in monitor.track(heartbeat_interval=0.001):
            if event is None:
                break

    set_when_done.set()  # shuts down websocket server
    await websocket_server_task
