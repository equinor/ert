import contextlib
import socket
import sys
import threading

import pytest

from ert.shared import port_handler


def test_find_available_port(unused_tcp_port):
    custom_range = range(unused_tcp_port, unused_tcp_port + 1)
    host, port, sock = port_handler.find_available_port(
        custom_range=custom_range, custom_host="127.0.0.1"
    )
    assert host is not None
    assert port is not None
    assert port in custom_range
    assert sock is not None
    assert sock.fileno() != -1


def test_find_available_port_forced(unused_tcp_port):
    custom_range = range(unused_tcp_port, unused_tcp_port)
    host, port, sock = port_handler.find_available_port(
        custom_range=custom_range, custom_host="127.0.0.1"
    )
    assert port == unused_tcp_port
    assert sock is not None
    assert sock.fileno() != -1


def test_invalid_host_name():
    invalid_host = "invalid_host"

    with pytest.raises(port_handler.InvalidHostException) as exc_info:
        port_handler.find_available_port(custom_host=invalid_host)

    assert (
        "Trying to bind socket with what looks "
        f"like an invalid hostname ({invalid_host})"
    ) in str(exc_info.value)


def test_get_family():
    family_inet6 = port_handler.get_family("::1")
    assert family_inet6 == socket.AF_INET6

    family_inet = port_handler.get_family("host:port")
    assert family_inet == socket.AF_INET

    family_inet = port_handler.get_family("host")
    assert family_inet == socket.AF_INET


def test_gc_closes_socket(unused_tcp_port):
    custom_range = range(unused_tcp_port, unused_tcp_port + 1)

    _, port, orig_sock = port_handler.find_available_port(
        custom_range=custom_range, custom_host="127.0.0.1"
    )
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1

    with pytest.raises(port_handler.NoPortsInRangeException):
        port_handler.find_available_port(
            custom_range=custom_range, custom_host="127.0.0.1"
        )

    with pytest.raises(port_handler.NoPortsInRangeException):
        port_handler.find_available_port(
            custom_range=custom_range,
            will_close_then_reopen_socket=True,
            custom_host="127.0.0.1",
        )

    orig_sock = None

    _, port, orig_sock = port_handler.find_available_port(
        custom_range=custom_range, custom_host="127.0.0.1"
    )
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
            conn, addr = sock.accept()
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


# Tests below checks results of trying to get a new socket on an
# already used port over permutations of 3 (boolean) parameters:
#
#     - mode when obtaining the first socket (default/reuse)
#     - activity on original socket or whether it is never used
#     - original socket live or closed
#
# The test-names encodes the permutation, the platform and finally
# whether subsequent calls to find_available_port() succeeds with
# default-mode and/or reuse-mode. For example:
#
#     test_def_active_close_macos_nok_ok
#
# means obtaining first socket in default-mode, activate it and
# then close it. On MacOS, trying to obtain it in default mode
# fails (nok) whereas obtaining it with reuse-flag succeeds (ok)
#
#
# Test identifier                           | mode  | activated | live
# ------------------------------------------+-------+-----------+------
# test_def_passive_live_nok_nok_close_ok_ok | def   | false     | both
#
# test_def_active_live_nok_nok              | def   | true      | true
# test_def_active_close_macos_nok_ok        | def   | true      | false
# test_def_active_close_linux_nok_nok       | def   | true      | false
#
# test_reuse_passive_live_macos_nok_nok     | reuse | false     | true
# test_reuse_passive_live_linux_nok_ok      | reuse | false     | true
# test_reuse_passive_close_ok_ok            | reuse | false     | false
# test_reuse_active_live_nok_nok            | reuse | true      | true
# test_reuse_active_close_nok_ok            | reuse | true      | false
#
#
# Note the behaviour of the first test: The recommended practice
# is to obtain the port/socket in default mode, keep the socket
# alive as long as the port is needed and provide dup() of the
# socket-object to other modules. If the other module cannot use
# an already bound socket, close the UN-ACTIVATED socket, give
# the port-number to the module and hope that no-one else grabs
# the port in the meantime. :)
#
# If you (for whatever obscure reason) activated the socket (i.e.
# some communication happened on the socket) and THEN provides
# the port-number to another module, you're on the last test and
# have to use reuse-mode when obtaining the first socket, and pray
# that the other module set SO_REUSEADDR before attempting to bind
# its socket.


def test_def_passive_live_nok_nok_close_ok_ok(unused_tcp_port):
    """
    Executive summary of this test

    1. the original socket is obtained in default, recommended mode
    2. no activity is triggered on the socket
    3. port is not closed but kept alive
    4. port can not be re-bound in any mode while socket-object is live
    5. port is closed
    6. after socket is closed the port can immediately be re-bound in any mode
    """
    custom_range = range(unused_tcp_port, unused_tcp_port + 1)

    # Opening original socket with will_close_then_reopen_socket=False
    _, port, orig_sock = port_handler.find_available_port(
        custom_range=custom_range, custom_host="127.0.0.1"
    )
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1

    # When the socket is kept open, this port can not be reused
    # with or without setting will_close_then_reopen_socket
    with pytest.raises(port_handler.NoPortsInRangeException):
        port_handler.find_available_port(
            custom_range=custom_range, custom_host="127.0.0.1"
        )

    with pytest.raises(port_handler.NoPortsInRangeException):
        port_handler.find_available_port(
            custom_range=custom_range,
            custom_host="127.0.0.1",
            will_close_then_reopen_socket=True,
        )

    orig_sock.close()

    # When we close the socket without actually having used it, it is
    # immediately reusable with or without setting will_close_then_reopen_socket
    _, port, sock = port_handler.find_available_port(
        custom_range=custom_range, custom_host="127.0.0.1"
    )
    assert port == unused_tcp_port
    assert sock is not None
    assert sock.fileno() != -1

    # we want to try again, so close it
    sock.close()

    _, port, sock = port_handler.find_available_port(
        custom_range=custom_range,
        custom_host="127.0.0.1",
        will_close_then_reopen_socket=True,
    )
    assert port == unused_tcp_port
    assert sock is not None
    assert sock.fileno() != -1


def test_reuse_active_close_nok_ok(unused_tcp_port):
    """
    Executive summary of this test

    1. the original socket is obtained with will_close_then_reopen_socket=True
    2. activity is triggered on the socket using a dummy-server/client
    3. socket is closed
    4. port can not be re-bound in default mode (TIME_WAIT?)...
    5. ... but can with will_close_then_reopen_socket=True (ignoring TIME_WAIT)
    """
    custom_range = range(unused_tcp_port, unused_tcp_port + 1)

    # Note: Setting will_close_then_reopen_socket=True on original socket
    host, port, orig_sock = port_handler.find_available_port(
        custom_range=custom_range,
        custom_host="127.0.0.1",
        will_close_then_reopen_socket=True,
    )
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1

    # Run a dummy-server to actually use the socket a little, then close it
    _simulate_server(host, port, orig_sock)
    orig_sock.close()

    # Using will_close_then_reopen_socket=False fails...
    with pytest.raises(port_handler.NoPortsInRangeException):
        port_handler.find_available_port(
            custom_range=custom_range, custom_host="127.0.0.1"
        )

    # ... but using will_close_then_reopen_socket=True succeeds
    _, port, sock = port_handler.find_available_port(
        custom_range=custom_range,
        custom_host="127.0.0.1",
        will_close_then_reopen_socket=True,
    )
    assert port == unused_tcp_port
    assert sock is not None
    assert sock.fileno() != -1


def test_reuse_active_live_nok_nok(unused_tcp_port):
    """
    Executive summary of this test

    1. the original socket is obtained with will_close_then_reopen_socket=True
    2. activity is triggered on the socket using a dummy-server/client
    3. socket is not closed but kept alive
    4. port can not be re-bound in default mode (TIME_WAIT?)...
    5. ... but can with will_close_then_reopen_socket=True (ignoring TIME_WAIT)
    """
    custom_range = range(unused_tcp_port, unused_tcp_port + 1)

    host, port, orig_sock = port_handler.find_available_port(
        custom_range=custom_range,
        custom_host="127.0.0.1",
        will_close_then_reopen_socket=True,
    )
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1

    # Run a dummy-server to actually use the socket a little, then close it
    _simulate_server(host, port, orig_sock)

    # Even with "will_close_then_reopen_socket"=True when obtaining original
    # socket, subsequent calls fails
    with pytest.raises(port_handler.NoPortsInRangeException):
        port_handler.find_available_port(
            custom_range=custom_range, custom_host="127.0.0.1"
        )

    with pytest.raises(port_handler.NoPortsInRangeException):
        _, port, sock = port_handler.find_available_port(
            custom_range=custom_range,
            custom_host="127.0.0.1",
            will_close_then_reopen_socket=True,
        )


def test_def_active_live_nok_nok(unused_tcp_port):
    """
    Executive summary of this test

    1. the original socket is obtained in default, recommended mode
    2. activity is triggered on the socket using a dummy-server/client
    3. socket is not closed but kept alive
    4. port can not be re-bound in any mode while socket-object is live
    """
    custom_range = range(unused_tcp_port, unused_tcp_port + 1)

    host, port, orig_sock = port_handler.find_available_port(
        custom_range=custom_range, custom_host="127.0.0.1"
    )
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1

    # Now, run a dummy-server to actually use the socket a little, do NOT close socket
    _simulate_server(host, port, orig_sock)

    # Immediately trying to bind to the same port fails...
    with pytest.raises(port_handler.NoPortsInRangeException):
        host, port, sock = port_handler.find_available_port(
            custom_range=custom_range, custom_host="127.0.0.1"
        )

    # ... also using will_close_then_reopen_socket=True
    with pytest.raises(port_handler.NoPortsInRangeException):
        host, port, sock = port_handler.find_available_port(
            custom_range=custom_range,
            custom_host="127.0.0.1",
            will_close_then_reopen_socket=True,
        )


@pytest.mark.skipif(
    not sys.platform.startswith("darwin"), reason="MacOS-specific socket behaviour"
)
def test_def_active_close_macos_nok_ok(unused_tcp_port):
    """
    Executive summary of this test

    1. the original socket is obtained in default, recommended mode
    2. activity is triggered on the socket using a dummy-server/client
    3. socket is closed
    4. after socket is closed the port can not be re-bound in
       default mode (TIME_WAIT?)...
    5. ...but it can be re-bound with will_close_then_reopen_socket=True
       (ignoring TIME_WAIT)
    """
    custom_range = range(unused_tcp_port, unused_tcp_port + 1)

    host, port, orig_sock = port_handler.find_available_port(
        custom_range=custom_range, custom_host="127.0.0.1"
    )
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1

    # Now, run a dummy-server to actually use the socket a little, then close it
    _simulate_server(host, port, orig_sock)
    orig_sock.close()

    # Immediately trying to bind to the same port fails
    with pytest.raises(port_handler.NoPortsInRangeException):
        _, _, sock = port_handler.find_available_port(
            custom_range=custom_range, custom_host="127.0.0.1"
        )

    # On MacOS, setting will_close_then_reopen_socket=True in subsequent calls allows
    # to reuse the port
    _, port, sock = port_handler.find_available_port(
        custom_range=custom_range,
        custom_host="127.0.0.1",
        will_close_then_reopen_socket=True,
    )
    assert port == unused_tcp_port
    assert sock is not None
    assert sock.fileno() != -1


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux-specific socket behaviour"
)
def test_def_active_close_linux_nok_nok(unused_tcp_port):
    """
    Executive summary of this test

    1. the original socket is obtained in default, recommended mode
    2. activity is triggered on the socket using a dummy-server/client
    3. socket is closed
    4. after socket is closed the port can not be re-bound in
       default mode (TIME_WAIT?)...
    5. ...nor with will_close_then_reopen_socket=True (not ignoring TIME_WAIT?)
    """
    custom_range = range(unused_tcp_port, unused_tcp_port + 1)

    host, port, orig_sock = port_handler.find_available_port(
        custom_range=custom_range, custom_host="127.0.0.1"
    )
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1

    # Now, run a dummy-server to actually use the socket a little, then close it
    _simulate_server(host, port, orig_sock)
    orig_sock.close()

    # Immediately trying to bind to the same port fails
    with pytest.raises(port_handler.NoPortsInRangeException):
        host, port, sock = port_handler.find_available_port(
            custom_range=custom_range, custom_host="127.0.0.1"
        )

    # On Linux, setting will_close_then_reopen_socket=True in subsequent calls do
    # NOT allow reusing the port in this case
    with pytest.raises(port_handler.NoPortsInRangeException):
        host, port, sock = port_handler.find_available_port(
            custom_range=custom_range,
            custom_host="127.0.0.1",
            will_close_then_reopen_socket=True,
        )


@pytest.mark.skipif(
    not sys.platform.startswith("darwin"), reason="MacOS-specific socket behaviour"
)
def test_reuse_passive_live_macos_nok_nok(unused_tcp_port):
    """
    Executive summary of this test

    1. the original socket is obtained with will_close_then_reopen_socket=True
    2. no activity is triggered on the socket
    3. the socket is not closed but kept alive
    4. port can not be re-bound in any mode
    """
    custom_range = range(unused_tcp_port, unused_tcp_port + 1)

    _, port, orig_sock = port_handler.find_available_port(
        custom_range=custom_range,
        custom_host="127.0.0.1",
        will_close_then_reopen_socket=True,
    )
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1

    # As long as the socket is kept alive this port can not be bound again...
    with pytest.raises(port_handler.NoPortsInRangeException):
        port_handler.find_available_port(
            custom_range=custom_range, custom_host="127.0.0.1"
        )

    # ... not even when setting will_close_then_reopen_socket=True
    with pytest.raises(port_handler.NoPortsInRangeException):
        port_handler.find_available_port(
            custom_range=custom_range,
            custom_host="127.0.0.1",
            will_close_then_reopen_socket=True,
        )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux-specific socket behaviour"
)
def test_reuse_passive_live_linux_nok_ok(unused_tcp_port):
    """
    Executive summary of this test

    1. the original socket is obtained with will_close_then_reopen_socket=True
    2. no activity is triggered on the socket
    3. the socket is not closed but kept alive
    4. port can not be re-bound in default mode...
    5. ... but can with will_close_then_reopen_socket=True
    """
    custom_range = range(unused_tcp_port, unused_tcp_port + 1)

    # Opening original socket with will_close_then_reopen_socket=True
    _, port, orig_sock = port_handler.find_available_port(
        custom_range=custom_range,
        custom_host="127.0.0.1",
        will_close_then_reopen_socket=True,
    )
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1

    # As long as the socket is kept alive this port can not be bound again...
    with pytest.raises(port_handler.NoPortsInRangeException):
        port_handler.find_available_port(
            custom_range=custom_range, custom_host="127.0.0.1"
        )

    # ... but on Linux the port can be re-bound by setting this flag!
    # This does not seem safe in a multi-user/-process environment!
    _, port, sock = port_handler.find_available_port(
        custom_range=custom_range,
        custom_host="127.0.0.1",
        will_close_then_reopen_socket=True,
    )
    assert port == unused_tcp_port
    assert sock is not None
    assert sock.fileno() != -1


def test_reuse_passive_close_ok_ok(unused_tcp_port):
    """
    Executive summary of this test

    1. the original socket is obtained with will_close_then_reopen_socket=True
    2. no activity is triggered on the socket
    3. the socket is closed
    4. port can be re-bound in any mode
    """
    custom_range = range(unused_tcp_port, unused_tcp_port + 1)

    _, port, orig_sock = port_handler.find_available_port(
        custom_range=custom_range,
        custom_host="127.0.0.1",
        will_close_then_reopen_socket=True,
    )
    assert port == unused_tcp_port
    assert orig_sock is not None
    assert orig_sock.fileno() != -1

    orig_sock.close()

    # When we close the socket without actually having used it, it is
    # immediately reusable with or without setting will_close_then_reopen_socket
    _, port, sock = port_handler.find_available_port(
        custom_range=custom_range, custom_host="127.0.0.1"
    )
    assert port == unused_tcp_port
    assert sock is not None
    assert sock.fileno() != -1

    # we want to try again, so close it
    sock.close()

    _, port, sock = port_handler.find_available_port(
        custom_range=custom_range,
        custom_host="127.0.0.1",
        will_close_then_reopen_socket=True,
    )
    assert port == unused_tcp_port
    assert sock is not None
    assert sock.fileno() != -1
