import pytest

from _ert.forward_model_runner.client import Client, ClientConnectionError
from tests.ert.utils import _mock_ws_task


async def test_invalid_server():
    port = 7777
    host = "localhost"
    url = f"ws://{host}:{port}"

    async with Client(url, max_retries=2, timeout_multiplier=2) as c1:
        with pytest.raises(ClientConnectionError):
            await c1.send("hei")


async def test_successful_sending(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []

    messages_c1 = ["test_1", "test_2", "test_3"]
    async with _mock_ws_task(host, unused_tcp_port, messages), Client(url) as c1:
        for msg in messages_c1:
            await c1.send(msg)

    for msg in messages_c1:
        assert msg in messages


async def test_retry(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []

    messages_c1 = ["test_1", "test_2", "test_3"]
    async with _mock_ws_task(host, unused_tcp_port, messages, delay_startup=2), Client(
        url, max_retries=2, timeout_multiplier=2
    ) as c1:
        for msg in messages_c1:
            await c1.send(msg)

    for msg in messages_c1:
        assert msg in messages
