from urllib.parse import urlparse

from ert.ensemble_evaluator.config import EvaluatorServerConfig


def test_load_config(unused_tcp_port):
    fixed_port = range(unused_tcp_port, unused_tcp_port)
    serv_config = EvaluatorServerConfig(
        custom_port_range=fixed_port,
        custom_host="127.0.0.1",
        localhost=False,
    )
    expected_host = "127.0.0.1"
    expected_port = unused_tcp_port
    expected_url = f"tcp://{expected_host}:{expected_port}"

    url = urlparse(serv_config.url)
    assert url.hostname == expected_host
    assert url.port == expected_port
    assert serv_config.url == expected_url
    assert serv_config.token is not None
    # TODO REFACTOR
    # sock = serv_config.get_socket()
    # assert sock is not None
    # assert not sock._closed
    # sock.close()

    # ee_config = EvaluatorServerConfig(
    #     custom_port_range=range(1024, 65535),
    #     custom_host="127.0.0.1",
    #     use_token=False,
    #     generate_cert=False,
    # )
    # sock = ee_config.get_socket()
    # assert sock is not None
    # assert not sock._closed
    # sock.close()
