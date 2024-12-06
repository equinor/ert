import asyncio

import pytest

from _ert.forward_model_runner.client import Client, ClientConnectionError
from tests.ert.utils import async_mock_zmq_server


@pytest.mark.integration_test
def test_invalid_server():
    port = 7777
    host = "localhost"
    url = f"tcp://{host}:{port}"

    with (
        pytest.raises(ClientConnectionError),
        Client(url, connection_timeout=1.0),
    ):
        pass


async def test_successful_sending(unused_tcp_port):
    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    messages = []
    server_started = asyncio.Event()

    server_task = asyncio.create_task(
        async_mock_zmq_server(messages, unused_tcp_port, server_started)
    )
    await server_started.wait()
    messages_c1 = ["test_1", "test_2", "test_3"]
    async with Client(url) as c1:
        await c1._send(messages_c1)

    await server_task

    for msg in messages_c1:
        assert msg in messages


async def test_retry(unused_tcp_port):
    pass
    # host = "localhost"
    # url = f"tcp://{host}:{unused_tcp_port}"
    # messages = []
    # server_started = asyncio.Event()

    # server_task = asyncio.create_task(
    #     async_mock_zmq_server(messages, unused_tcp_port, server_started)
    # )

    # messages_c1 = ["test_1", "test_2", "test_3"]

    # TODO write test for retry!
