import pytest

from ert.shared.ensemble_evaluator.client import Client


def test_invalid_server():
    port = 7777
    host = "localhost"
    url = f"ws://{host}:{port}"

    with Client(url, max_retries=2, timeout_multiplier=2) as c1:
        with pytest.raises((ConnectionRefusedError, OSError)):
            c1.send("hei")


def test_successful_sending(unused_tcp_port, mock_ws_thread):
    messages = []
    with mock_ws_thread("localhost", unused_tcp_port, messages):
        messages_c1 = ["test_1", "test_2", "test_3"]

        with Client(f"ws://localhost:{unused_tcp_port}") as c1:
            for msg in messages_c1:
                c1.send(msg)

    for msg in messages_c1:
        assert msg in messages


def test_retry(unused_tcp_port, mock_ws_thread):
    messages = []
    with mock_ws_thread("localhost", unused_tcp_port, messages, delay_startup=2):
        messages_c1 = ["test_1", "test_2", "test_3"]

        with Client(
            f"ws://localhost:{unused_tcp_port}", max_retries=2, timeout_multiplier=2
        ) as c1:
            for msg in messages_c1:
                c1.send(msg)

    for msg in messages_c1:
        assert msg in messages
