import queue

import pytest

from _ert.forward_model_runner.client import Client, ClientConnectionError
from tests.ert.utils import mock_zmq_thread


@pytest.mark.integration_test
def test_invalid_server():
    port = 7777
    host = "localhost"
    url = f"tcp://{host}:{port}"

    with (
        pytest.raises(ClientConnectionError),
        Client(url, ack_timeout=1.0),
    ):
        pass


def test_successful_sending(unused_tcp_port):
    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    messages = []
    with mock_zmq_thread(unused_tcp_port, messages):
        messages_c1 = ["test_1", "test_2", "test_3"]
        with Client(url) as c1:
            for message in messages_c1:
                c1.send(message)

    for msg in messages_c1:
        assert msg in messages


def test_retry(unused_tcp_port):
    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    messages = []
    signal_queue = queue.Queue()
    signal_queue.put(2)
    client_connection_error_set = False
    with mock_zmq_thread(unused_tcp_port, messages, signal_queue):
        messages_c1 = ["test_1", "test_2", "test_3"]
        with Client(url, ack_timeout=1) as c1:
            for message in messages_c1:
                try:
                    c1.send(message, retries=2)
                except ClientConnectionError:
                    client_connection_error_set = True
                    signal_queue.put(0)
    assert client_connection_error_set
    assert messages.count("test_1") == 3
    assert messages.count("test_2") == 1
    assert messages.count("test_3") == 1
