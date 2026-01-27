import contextlib
import socket
import threading

import psutil
import pytest

from ert.shared import find_available_socket, get_machine_name
from ert.shared.net_utils import (
    NoPortsInRangeException,
    get_family,
)


def check_port_open(host: str, port: int) -> bool:
    try:
        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.connect((host, port))
    except TimeoutError:
        print(f"Port {port} is not open on {host} (connection timed out)")
        return False
    except ConnectionRefusedError:
        print(f"Port {port} is not open on {host} (connection refused)")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    finally:
        sock.close()

    return True


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

    # Avoid possibility of flakyness in code paths not relevant
    # for this test:
    mocker.patch("socket.gethostname", return_value=None)
    mocker.patch("socket.gethostbyname", return_value=None)
    mocker.patch("dns.reversename.from_address", return_value=None)

    # This call is what this test wants to test:
    mocker.patch("dns.resolver.resolve", return_value=ptr_records)
    get_machine_name.cache_clear()
    # ASSERT the returned name
    assert get_machine_name() == expected_resolved_name
    get_machine_name.cache_clear()

    # Shuffle the the list and try again:
    ptr_records.reverse()
    mocker.patch("dns.resolver.resolve", return_value=ptr_records)

    # ASSERT that we still get the same name
    assert get_machine_name() == expected_resolved_name


def test_find_available_socket(unused_tcp_port):
    port_range = range(unused_tcp_port, unused_tcp_port + 1)
    sock = find_available_socket(port_range=port_range, host="127.0.0.1")
    (
        host,
        port,
    ) = sock.getsockname()
    assert host is not None
    assert port is not None
    assert port in port_range
    assert sock is not None
    assert sock.fileno() != -1


def test_find_available_socket_forced(unused_tcp_port):
    port_range = range(unused_tcp_port, unused_tcp_port)
    sock = find_available_socket(port_range=port_range, host="127.0.0.1")
    (
        _,
        port,
    ) = sock.getsockname()
    assert port == unused_tcp_port
    assert sock is not None
    assert sock.fileno() != -1


def test_that_find_available_socket_host_name_is_irrelevant():
    # We bind to all interfaces on machine
    irrelevant_host = "this_is_not_a_valid_host_name"
    sock = find_available_socket(host=irrelevant_host)
    assert sock.getsockname()[0] == "0.0.0.0"


def test_that_find_available_socket_ipv6_binds_local():
    sock = find_available_socket(host="::1")
    assert sock.getsockname()[0] == "::1"


def test_that_find_available_socket_raises_on_invalid_ipv6_host():
    invalid_host = "0:0:0:0:0:0:0:2"
    with pytest.raises(OSError, match="Unknown `OSError` while binding port"):
        find_available_socket(invalid_host)


def test_that_find_available_socket_default_params_binds_all_interfaces():
    # IPV4 only: Check that when listening on the bound socket,
    # all relevant interfaces is listening
    sock = find_available_socket()
    assert sock.getsockname()[0] == "0.0.0.0"
    sock.listen()

    net_if_stats = psutil.net_if_stats()
    previous_interface = ""
    ipv4_interfaces = []
    for interface, snicaddrs in psutil.net_if_addrs().items():
        for snicaddr in snicaddrs:
            if snicaddr.family == socket.AF_INET and (
                snicaddr.address == "127.0.0.1" or snicaddr.broadcast
            ):  # include loopback and broadcast addresses
                if previous_interface != interface and net_if_stats[interface].isup:
                    ipv4_interfaces.append(snicaddr.address)
                previous_interface = interface

    for ipv4_interface in ipv4_interfaces:
        print(f"Checking interface {ipv4_interface}")
        assert check_port_open(ipv4_interface, sock.getsockname()[1]), (
            f"Port {sock.getsockname()[1]} not open on {ipv4_interface}"
        )


def test_get_family():
    family_inet6 = get_family("::1")
    assert family_inet6 == socket.AF_INET6

    family_inet = get_family("host:port")
    assert family_inet == socket.AF_INET

    family_inet = get_family("host")
    assert family_inet == socket.AF_INET


def test_gc_closes_socket(unused_tcp_port):
    port_range = range(unused_tcp_port, unused_tcp_port + 1)

    orig_sock = find_available_socket(port_range=port_range, host="127.0.0.1")
    _, port = orig_sock.getsockname()
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1

    with pytest.raises(NoPortsInRangeException):
        find_available_socket(port_range=port_range, host="127.0.0.1")

    orig_sock = None

    orig_sock = find_available_socket(port_range=port_range, host="127.0.0.1")
    _, port = orig_sock.getsockname()
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1


def _simulate_server(host, port, sock: socket.socket):
    """
    This seems to be necessary to demonstrate TIME_WAIT on sockets.
    Just opening and closing sockets doesn't really activate underlying sockets.
    This is also more similar of how real applications might behave.
    """
    ready_event = threading.Event()

    class ServerThread(threading.Thread):
        def run(self):
            self.port = port
            sock.listen()
            ready_event.set()
            conn, _ = sock.accept()
            with contextlib.suppress(Exception):
                self.data = conn.recv(1024).decode()
                conn.sendall(b"Who's there?")

    dummy_server = ServerThread()
    dummy_server.start()
    ready_event.wait()

    client_socket = socket.socket()
    client_socket.connect((host, port))
    client_socket.sendall(b"Hi there")
    assert client_socket.recv(1024).decode() == "Who's there?"
    dummy_server.join()
    assert getattr(dummy_server, "port", None) == port
    assert getattr(dummy_server, "data", None) == "Hi there"


def test_socket_can_rebind_if_never_used(unused_tcp_port):
    """
    1. the original socket is obtained
    2. no activity is triggered on the socket
    3. port is not closed but kept alive
    4. port can not be re-bound in any mode while socket-object is live
    5. port is closed
    6. after socket is closed the port can immediately be re-bound in any mode
    """
    port_range = range(unused_tcp_port, unused_tcp_port + 1)

    # Opening original socket with will_close_then_reopen_socket=False
    orig_sock = find_available_socket(port_range=port_range, host="127.0.0.1")
    _, port = orig_sock.getsockname()
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1

    # When the socket is kept open
    with pytest.raises(NoPortsInRangeException):
        find_available_socket(port_range=port_range, host="127.0.0.1")

    orig_sock.close()

    # When we close the socket without actually having used it, it is
    # immediately reusable
    sock = find_available_socket(port_range=port_range, host="127.0.0.1")
    _, port = sock.getsockname()
    assert port == unused_tcp_port
    assert sock is not None
    assert sock.fileno() != -1

    # we want to try again, so close it
    sock.close()

    sock = find_available_socket(port_range=port_range, host="127.0.0.1")
    _, port = sock.getsockname()
    assert port == unused_tcp_port
    assert sock is not None
    assert sock.fileno() != -1


def test_socket_can_not_rebind_if_open(unused_tcp_port):
    """
    1. the original socket is obtained
    2. activity is triggered on the socket using a dummy-server/client
    3. socket is not closed but kept alive
    4. port can not be re-bound while socket-object is live
    """
    port_range = range(unused_tcp_port, unused_tcp_port + 1)

    orig_sock = find_available_socket(port_range=port_range, host="127.0.0.1")
    host, port = orig_sock.getsockname()
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1

    # Now, run a dummy-server to actually use the socket a little, do NOT close socket
    _simulate_server(host, port, orig_sock)

    # Immediately trying to bind to the same port fails...
    with pytest.raises(NoPortsInRangeException):
        find_available_socket(port_range=port_range, host="127.0.0.1")


@pytest.mark.slow
def test_socket_can_not_rebind_immediately_after_close_if_used(unused_tcp_port):
    """
    1. the original socket is obtained
    2. activity is triggered on the socket using a dummy-server/client
    3. socket is closed
    4. after socket is closed the port can not be re-bound immediately due to TIME_WAIT
    """
    port_range = range(unused_tcp_port, unused_tcp_port + 1)

    orig_sock = find_available_socket(port_range=port_range, host="127.0.0.1")
    host, port = orig_sock.getsockname()
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1

    # Now, run a dummy-server to actually use the socket a little, then close it
    _simulate_server(host, port, orig_sock)
    orig_sock.close()

    # Immediately trying to bind to the same port fails
    with pytest.raises(NoPortsInRangeException):
        find_available_socket(port_range=port_range, host="127.0.0.1")
