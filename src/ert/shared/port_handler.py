import logging
import random
import socket
from typing import Optional, Tuple


class PortAlreadyInUseException(Exception):
    pass


class NoPortsInRangeException(Exception):
    pass


class InvalidHostException(Exception):
    pass


logger = logging.getLogger(__name__)


def find_available_port(
    custom_host: Optional[str] = None,
    custom_range: Optional[range] = None,
    will_close_then_reopen_socket: bool = False,
) -> Tuple[str, int, socket.socket]:
    """
    The default and recommended approach here is to return a bound socket to the
    caller, requiring the caller to keep the socket-object alive as long as the
    port is needed.

    If the caller for some reason closes the returned socket there is no guarantee
    that it can bind again to the same port for two reasons:
    1) the underlying socket can be in TIME_WAIT meaning that it is closed
    but port is not ready to be re-bound yet, and 2) some other process managed to
    bind the port before the original caller gets around to re-bind.

    Thus, we expect clients calling find_available_port() to keep the returned
    socket-object alive and open as long as the port is needed. If a socket-object
    is passed to other modules like for example a websocket-server, use dup() to
    obtain a new Python socket-object bound to the same underlying socket (and hence
    the same port). That way, even if the other module closes its given socket-
    object, the port is still reserved and bound by the original socket-object.

    See e.g. implementation and comments in EvaluatorServerConfig

    However, our PrefectEnsemble integrates with Dask and DaskScheduler which only
    accepts an integer port-argument, not a bound socket-object or even a port-range.
    For this reason it is believed that we must allow the close-and-rebind  -pattern,
    i.e. the SO_REUSEADDR flag may have to be set before binding the socket. If
    the client really wants this behaviour, specify the argument

        will_close_then_reopen_socket = true

    in the call, but be aware that issue (2) above becomes a lot more likely.
    Additionally, quite some unexpected behaviour is going on when this flag is used,
    as demonstrated in test_port_handler.py. In particular, it looks like on Linux,
    a port can be re-bound even if there is already a live socket bound to it.

    So use this flag with extreme caution and only when in the hour of utmost need!
    Ideally it should be removed.
    """
    current_host = custom_host if custom_host is not None else _get_ip_address()
    current_range = (
        custom_range if custom_range is not None else range(51820, 51840 + 1)
    )

    if current_range.start == current_range.stop:
        ports = list(range(current_range.start, current_range.stop + 1))
    else:
        ports = list(range(current_range.start, current_range.stop))

    random.shuffle(ports)
    for port in ports:
        try:
            return (
                current_host,
                port,
                _bind_socket(
                    host=current_host,
                    port=port,
                    will_close_then_reopen_socket=will_close_then_reopen_socket,
                ),
            )
        except PortAlreadyInUseException:
            continue

    raise NoPortsInRangeException(f"No available ports in range {current_range}.")


def _bind_socket(
    host: str, port: int, will_close_then_reopen_socket: bool = False
) -> socket.socket:
    try:
        family = get_family(host=host)
        sock = socket.socket(family=family, type=socket.SOCK_STREAM)

        # Setting flags like SO_REUSEADDR and/or SO_REUSEPORT may have
        # undesirable side-effects but we allow it if caller insists. Refer to
        # comment on find_available_port()
        #
        # See e.g.  https://stackoverflow.com/a/14388707 for an extensive
        # explanation of these flags, in particular the part about TIME_WAIT

        if will_close_then_reopen_socket:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)

        sock.bind((host, port))
        return sock
    except socket.gaierror as err_info:
        raise InvalidHostException(
            f"Trying to bind socket with what looks like "
            f"an invalid hostname ({host}). "
            f"Actual "
            f"error msg is: {err_info.strerror}"
        )
    except OSError as err_info:
        if err_info.errno in (48, 98):
            raise PortAlreadyInUseException(f"Port {port} already in use.")
        raise Exception(
            f"Unknown `OSError` while binding port {port}. Actual "
            f"error msg is: {err_info.strerror}"
        )


def get_family_for_localhost() -> socket.AddressFamily:
    # pylint: disable=no-member
    # (false positive on socket.AddressFamily)
    return get_family(_get_ip_address())


def get_family(host: str) -> socket.AddressFamily:
    # pylint: disable=no-member
    # (false positive on socket.AddressFamily)
    try:
        socket.inet_pton(socket.AF_INET6, host)
        return socket.AF_INET6
    except socket.error:
        return socket.AF_INET


# See https://stackoverflow.com/a/28950776
def _get_ip_address() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        # try pinging a reserved, internal address in order
        # to determine IP representing the default route
        s.connect(("10.255.255.255", 1))
        retval = s.getsockname()[0]
    except BaseException:  # pylint: disable=broad-except
        logger.warning("Cannot determine ip-address. Fallback to localhost...")
        retval = "127.0.0.1"
    finally:
        s.close()
    logger.debug(f"ip-address: {retval}")
    return retval
