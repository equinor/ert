from urllib.parse import urlparse

from ert.ensemble_evaluator.config import EvaluatorServerConfig


def test_ensemble_evaluator_config_tcp_protocol(unused_tcp_port):
    fixed_port = range(unused_tcp_port, unused_tcp_port)
    serv_config = EvaluatorServerConfig(
        custom_port_range=fixed_port,
        custom_host="127.0.0.1",
        use_ipc_protocol=False,
    )
    expected_host = "127.0.0.1"
    expected_port = unused_tcp_port
    expected_url = f"tcp://{expected_host}:{expected_port}"

    url = urlparse(serv_config.url)
    assert url.hostname == expected_host
    assert url.port == expected_port
    assert serv_config.url == expected_url
    assert serv_config.token is not None
    assert serv_config.server_public_key is not None
    assert serv_config.server_secret_key is not None
    sock = serv_config.get_socket()
    assert sock is not None
    assert not sock._closed
    sock.close()


def test_ensemble_evaluator_config_ipc_protocol():
    serv_config = EvaluatorServerConfig(use_ipc_protocol=True, use_token=False)

    assert serv_config.url.startswith("ipc:///tmp/socket-")
    assert serv_config.token is None
    assert serv_config.server_public_key is None
    assert serv_config.server_secret_key is None
    assert serv_config.get_socket() is None
