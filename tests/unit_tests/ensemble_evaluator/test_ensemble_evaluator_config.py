from dns import resolver  # noqa # pylint: disable=unused-import

from ert.ensemble_evaluator.config import EvaluatorServerConfig, get_machine_name


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

    assert serv_config.host == expected_host
    assert serv_config.port == expected_port
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


def test_that_get_machine_name_is_predictive(mocker):
    """For ip addresses with multiple PTR records we must ensure
    that get_machine_name() is predictive to avoid mismatch for SSL certificates.

    The order DNS servers respond to reverse DNS lookups for such hosts is not
    defined."""

    # GIVEN that reverse DNS resolution results in two names (in random order):
    ptr_records = ["barfoo01.internaldomain.barf.", "foobar01.equinor.com."]

    # It is important that get_machine_name() is predictive for each
    # invocation, not how it attains predictiveness. Currently the PTR records
    # are sorted and the first element is returned, but that should be regarded
    # as an implementation detail.
    expected_resolved_name = ptr_records[0].rstrip(".")

    mocker.patch("dns.resolver.resolve", return_value=ptr_records)

    # ASSERT the returned name
    assert get_machine_name() == expected_resolved_name

    # Shuffle the the list and try again:
    ptr_records.reverse()
    mocker.patch("dns.resolver.resolve", return_value=ptr_records)

    # ASSERT that we still get the same name
    assert get_machine_name() == expected_resolved_name
    # assert get_machine_name() in [expected_resolved_name, "localhost"]

    # (Flakyness has been seen to give exceptions, which will be returned
    # as "localhost". That situation is ignored in this test)
