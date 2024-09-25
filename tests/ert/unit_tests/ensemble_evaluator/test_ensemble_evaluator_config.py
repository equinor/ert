from urllib.parse import urlparse

from ert.ensemble_evaluator.config import EvaluatorServerConfig


def test_load_config(unused_tcp_port):
    fixed_port = range(unused_tcp_port, unused_tcp_port)
    serv_config = EvaluatorServerConfig(
        custom_port_range=fixed_port,
        custom_host="127.0.0.1",
    )
    expected_host = "127.0.0.1"
    expected_port = unused_tcp_port
    expected_url = f"wss://{expected_host}:{expected_port}"
    expected_client_uri = f"{expected_url}/client"
    expected_dispatch_uri = f"{expected_url}/dispatch"

    url = urlparse(serv_config.url)
    assert url.hostname == expected_host
    assert url.port == expected_port
    assert serv_config.url == expected_url
    assert serv_config.client_uri == expected_client_uri
    assert serv_config.dispatch_uri == expected_dispatch_uri
    assert serv_config.token is not None
    assert serv_config.cert is not None
    sock = serv_config.get_socket()
    assert sock is not None
    assert not sock._closed
    sock.close()

    ee_config = EvaluatorServerConfig(
        custom_port_range=range(1024, 65535),
        custom_host="127.0.0.1",
        use_token=False,
        generate_cert=False,
    )
    sock = ee_config.get_socket()
    assert sock is not None
    assert not sock._closed
    sock.close()
