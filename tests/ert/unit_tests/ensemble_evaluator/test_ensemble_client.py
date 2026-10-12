import pytest

from _ert.forward_model_runner.client import Client, ClientConnectionError
from tests.ert.utils import MockZMQServer


async def test_invalid_server(monkeypatch):
    port = 7777
    host = "localhost"
    url = f"tcp://{host}:{port}"

    monkeypatch.setattr(Client, "DEFAULT_MAX_RETRIES", 0)

    with pytest.raises(ClientConnectionError):
        async with Client(url, ack_timeout=0.01):
            pass


async def test_successful_sending():
    messages = ["test_1", "test_2", "test_3"]
    async with MockZMQServer() as mock_server, Client(mock_server.uri) as client:
        for message in messages:
            await client.send(message)

    for msg in messages:
        assert msg in mock_server.messages


@pytest.mark.slow
async def test_retry():
    client_connection_error_set = False
    messages = ["test_1", "test_2", "test_3"]
    async with (
        MockZMQServer(dont_ack_messages=True) as mock_server,
        Client(mock_server.uri, ack_timeout=0.5) as client,
    ):
        for message in messages:
            try:
                await client.send(message, retries=1)
            except ClientConnectionError:
                client_connection_error_set = True
                mock_server.ack_messages = True
    assert client_connection_error_set
    assert mock_server.messages.count("test_1") == 2
    assert mock_server.messages.count("test_2") == 1
    assert mock_server.messages.count("test_3") == 1
