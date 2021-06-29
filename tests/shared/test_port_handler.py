import socket

import pytest

from ert_shared import port_handler


def test_find_available_port():
    custom_range = range(50000, 50001)
    host, port = port_handler.find_available_port(custom_range=custom_range)
    sock = port_handler.get_socket(host, port)
    assert host is not None
    assert port is not None
    assert port in custom_range
    assert sock is not None
    assert sock.fileno() != -1


def test_find_available_port_forced(unused_tcp_port):
    custom_range = range(unused_tcp_port, unused_tcp_port)
    host, port = port_handler.find_available_port(custom_range=custom_range)
    assert port == unused_tcp_port

    sock = port_handler.get_socket(host, port)
    assert sock is not None
    assert sock.fileno() != -1


def test_no_more_ports_in_range():
    custom_range = range(50000, 50001)
    host, port = port_handler.find_available_port(custom_range=custom_range)
    sock = port_handler.get_socket(host, port)
    assert sock is not None
    assert sock.fileno() != -1

    with pytest.raises(port_handler.NoPortsInRangeException) as exc_info:
        port_handler.find_available_port(custom_range=custom_range)


def test_invalid_host_name():
    invalid_host = "invalid_host"

    with pytest.raises(port_handler.InvalidHostException) as exc_info:
        port_handler.find_available_port(custom_host=invalid_host)

    assert (
        f"Trying to bind socket with what looks like an invalid hostname ({invalid_host})"
        in str(exc_info.value)
    )


def test_get_family():
    family_inet6 = port_handler.get_family("::1")
    assert family_inet6 == socket.AF_INET6

    family_inet = port_handler.get_family("host:port")
    assert family_inet == socket.AF_INET

    family_inet = port_handler.get_family("host")
    assert family_inet == socket.AF_INET
