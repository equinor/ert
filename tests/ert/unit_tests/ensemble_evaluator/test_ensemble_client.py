import pytest

from _ert.forward_model_runner.client import Client, ClientConnectionError
from tests.ert.utils import MockZMQServer


@pytest.mark.integration_test
async def test_invalid_server():
    port = 7777
    host = "localhost"
    url = f"tcp://{host}:{port}"

    with pytest.raises(ClientConnectionError):
        async with Client(url, ack_timeout=1.0):
            pass


async def test_successful_sending(unused_tcp_port):
    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    messages_c1 = ["test_1", "test_2", "test_3"]
    async with MockZMQServer(unused_tcp_port) as mock_server, Client(url) as c1:
        for message in messages_c1:
            await c1.send(message)

    for msg in messages_c1:
        assert msg in mock_server.messages


async def test_retry(unused_tcp_port):
    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    client_connection_error_set = False
    messages_c1 = ["test_1", "test_2", "test_3"]
    async with (
        MockZMQServer(unused_tcp_port, signal=2) as mock_server,
        Client(url, ack_timeout=0.5) as c1,
    ):
        for message in messages_c1:
            try:
                await c1.send(message, retries=1)
            except ClientConnectionError:
                client_connection_error_set = True
                mock_server.signal(0)
    assert client_connection_error_set
    assert mock_server.messages.count("test_1") == 2
    assert mock_server.messages.count("test_2") == 1
    assert mock_server.messages.count("test_3") == 1
