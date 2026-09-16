import logging
import random
import socket
import threading
from collections.abc import Callable
from functools import lru_cache

from dns import exception, resolver, reversename

from ert.shared.constants import PORT_RANGE


class PortAlreadyInUseException(Exception):
    pass


class NoPortsInRangeException(Exception):
    pass


class InvalidHostException(Exception):
    pass


class ResolverStalled(Exception):
    """Raised when a blocking OS resolver call exceeds our patience budget."""


logger = logging.getLogger(__name__)

# socket.gethostbyname()/socket.getfqdn() defer to the OS resolver and have no
# built-in timeout. On some CI runners (observed on GitHub Actions macOS
# runners) DNS resolution can stall for a long time without raising, which
# risks stalling the storage server boot sequence beyond its own timeout
# budget.
GETHOSTBYNAME_TIMEOUT_SECONDS = 3.0
GETFQDN_TIMEOUT_SECONDS = 3.0


def _run_with_timeout[T](func: Callable[[], T], timeout: float) -> T:
    """Runs `func` in a background thread and raises `ResolverStalled` if it
    does not complete within `timeout` seconds. Otherwise returns its result,
    or re-raises whatever exception `func` raised.

    There is no way to cancel a blocked OS-level resolver call, so on a stall
    the lookup thread is left running in the background (as a daemon).
    """
    result: list[T] = []
    raised_exception: list[BaseException] = []

    def _target() -> None:
        try:
            result.append(func())
        except BaseException as exc:
            raised_exception.append(exc)

    lookup_thread = threading.Thread(target=_target, daemon=True)
    lookup_thread.start()
    lookup_thread.join(timeout)

    if lookup_thread.is_alive():
        raise ResolverStalled(f"{func} did not complete within {timeout}s")
    if raised_exception:
        raise raised_exception[0]
    return result[0]


def get_fqdn_with_timeout(timeout: float = GETFQDN_TIMEOUT_SECONDS) -> str:
    """Returns socket.getfqdn(), but never blocks longer than `timeout` seconds.

    Falls back to socket.gethostname() if the lookup does not complete in time.
    """
    try:
        return _run_with_timeout(socket.getfqdn, timeout)
    except ResolverStalled:
        logger.warning(
            f"socket.getfqdn() did not resolve within {timeout}s, "
            "falling back to socket.gethostname()"
        )
        return socket.gethostname()


@lru_cache
def get_machine_name() -> str:
    """Returns a name that can be used to identify this machine in a network
    A fully qualified domain name is returned if available. Otherwise returns
    the string `localhost`
    """
    hostname = socket.gethostname()
    try:
        # We need the ip-address to perform a reverse lookup to deal with
        # differences in how the clusters are getting their fqdn's
        ip_addr = _run_with_timeout(
            lambda: socket.gethostbyname(hostname), GETHOSTBYNAME_TIMEOUT_SECONDS
        )
        reverse_name = reversename.from_address(ip_addr)
        resolved_hosts = [
            str(ptr_record).rstrip(".")
            for ptr_record in resolver.resolve(reverse_name, "PTR")
        ]
        resolved_hosts.sort()
        return resolved_hosts[0]
    except (
        resolver.NXDOMAIN,
        exception.Timeout,
        resolver.NoResolverConfiguration,
        ResolverStalled,
    ):
        # If local address and reverse lookup not working - fallback
        # to socket fqdn which are using /etc/hosts to retrieve this name
        return get_fqdn_with_timeout()
    except (socket.gaierror, exception.DNSException):
        return "localhost"


def find_available_socket(
    host: str | None = None,
    port_range: range = range(PORT_RANGE[0], PORT_RANGE[1]),
) -> socket.socket:
    """
    The default and recommended approach here is to return a bound socket to the
    caller, requiring the caller to keep the socket-object alive as long as the
    port is needed.

    If the caller for some reason closes the returned socket there is no guarantee
    that it can bind again to the same port for the following reason:
    The underlying socket can be in TIME_WAIT meaning that it is closed
    but port is not ready to be re-bound yet, and 2) some other process managed to
    bind the port before the original caller gets around to re-bind.

    Thus, we expect clients calling find_available_socket() to keep the returned
    socket-object alive and open as long as the port is needed. If a socket-object
    is passed to other modules like for example a websocket-server, use dup() to
    obtain a new Python socket-object bound to the same underlying socket (and hence
    the same port). That way, even if the other module closes its given socket-
    object, the port is still reserved and bound by the original socket-object.

    See e.g. implementation and comments in EvaluatorServerConfig
    """
    current_host = host if host is not None else get_ip_address()

    if port_range.start == port_range.stop:
        ports = list(range(port_range.start, port_range.stop + 1))
    else:
        ports = list(range(port_range.start, port_range.stop))

    random.shuffle(ports)
    for port in ports:
        try:
            return _bind_socket(
                host=current_host,
                port=port,
            )
        except PortAlreadyInUseException:
            continue

    raise NoPortsInRangeException(
        f"No available ports in range {port_range}. "
        "Perhaps you are running too many instances of Ert."
    )


def _bind_socket(host: str, port: int) -> socket.socket:
    """Binds a socket to the given port.
    NOTE: The host is only used to determine if we should bind to
    all interfaces (ipv4) or a specific interface (ipv6).
    """

    try:  # ruff: ignore[too-many-statements-in-try-clause]
        family = get_family(host=host)
        sock = socket.socket(family=family, type=socket.SOCK_STREAM)
        if family == socket.AF_INET6:
            sock.bind((host, port))
        else:
            sock.bind(
                ("", port)
            )  # Bind to all interfaces when IPV4, https://docs.python.org/3/library/socket.html#socket-families
    except socket.gaierror as err_info:
        raise InvalidHostException(
            f"Trying to bind socket with what looks like "
            f"an invalid hostname (IP) ({host}). "
            f"Actual "
            f"error msg is: {err_info.strerror}"
        ) from err_info
    except OSError as err_info:
        if err_info.errno in {48, 98}:
            raise PortAlreadyInUseException(
                f"Port {port} already in use."
            ) from err_info
        raise OSError(f"Unknown `OSError` while binding port {port}") from err_info
    else:
        return sock


def get_family(host: str) -> socket.AddressFamily:
    try:
        socket.inet_pton(socket.AF_INET6, host)
    except OSError:
        return socket.AF_INET
    else:
        return socket.AF_INET6


# See https://stackoverflow.com/a/28950776
def get_ip_address() -> str:
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.settimeout(0)
            # try pinging a reserved, internal address in order
            # to determine IP representing the default route
            s.connect(("10.255.255.255", 1))
            address = s.getsockname()[0]
        finally:
            s.close()
    except BaseException:
        logger.warning("Cannot determine ip-address. Falling back to localhost.")
        address = "127.0.0.1"
    logger.debug(f"ip-address: {address}")
    return address
