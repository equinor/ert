from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig, _get_ip_address


def test_load_config(unused_tcp_port):
    serv_config = EvaluatorServerConfig(unused_tcp_port)
    expected_host = _get_ip_address()
    expected_port = unused_tcp_port
    expected_url = f"ws://{expected_host}:{expected_port}"
    expected_client_uri = f"{expected_url}/client"
    expected_dispatch_uri = f"{expected_url}/dispatch"

    assert serv_config.host == expected_host
    assert serv_config.port == expected_port
    assert serv_config.url == expected_url
    assert serv_config.client_uri == expected_client_uri
    assert serv_config.dispatch_uri == expected_dispatch_uri
    sock = serv_config.get_socket()
    assert sock is not None
    assert not sock._closed
    sock.close()

    ee_config = EvaluatorServerConfig()
    assert ee_config.port in range(51820, 51840)
    sock = ee_config.get_socket()
    assert sock is not None
    assert not sock._closed
    sock.close()
