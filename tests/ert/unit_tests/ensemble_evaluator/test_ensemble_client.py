import asyncio

import pytest

import _ert.forward_model_runner.client
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


@pytest.mark.integration_test
async def test_retry():
    client_connection_error_set = False
    messages = ["test_1", "test_2", "test_3"]
    async with (
        MockZMQServer(signal=2) as mock_server,
        Client(mock_server.uri, ack_timeout=0.5) as client,
    ):
        for message in messages:
            try:
                await client.send(message, retries=1)
            except ClientConnectionError:
                client_connection_error_set = True
                mock_server.signal(0)
    assert client_connection_error_set
    assert mock_server.messages.count("test_1") == 2
    assert mock_server.messages.count("test_2") == 1
    assert mock_server.messages.count("test_3") == 1


async def test_reconnect_when_missing_heartbeat(monkeypatch):
    async with MockZMQServer(signal=3) as mock_server:
        monkeypatch.setattr(_ert.forward_model_runner.client, "HEARTBEAT_TIMEOUT", 0.01)
        async with Client(mock_server.uri) as client:
            await client.send("start", retries=1)

            await mock_server.do_heartbeat()
            await asyncio.sleep(0.1)
            await mock_server.do_heartbeat()
            await client.send("stop", retries=1)

    # the client should be disconnected
    assert len(mock_server.dealers) == 0

    # when reconnection happens CONNECT message is sent again
    assert mock_server.messages.count("CONNECT") == 2
    assert mock_server.messages.count("DISCONNECT") == 1
    assert "start" in mock_server.messages
    assert "stop" in mock_server.messages
