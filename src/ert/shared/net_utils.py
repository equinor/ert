import logging
import random
import socket

from dns import exception, resolver, reversename


class PortAlreadyInUseException(Exception):
    pass


class NoPortsInRangeException(Exception):
    pass


class InvalidHostException(Exception):
    pass


logger = logging.getLogger(__name__)


def get_machine_name() -> str:
    """Returns a name that can be used to identify this machine in a network
    A fully qualified domain name is returned if available. Otherwise returns
    the string `localhost`
    """
    hostname = socket.gethostname()
    try:
        # We need the ip-address to perform a reverse lookup to deal with
        # differences in how the clusters are getting their fqdn's
        ip_addr = socket.gethostbyname(hostname)
        reverse_name = reversename.from_address(ip_addr)
        resolved_hosts = [
            str(ptr_record).rstrip(".")
            for ptr_record in resolver.resolve(reverse_name, "PTR")
        ]
        resolved_hosts.sort()
        return resolved_hosts[0]
    except (resolver.NXDOMAIN, exception.Timeout):
        # If local address and reverse lookup not working - fallback
        # to socket fqdn which are using /etc/hosts to retrieve this name
        return socket.getfqdn()
    except (socket.gaierror, exception.DNSException):
        return "localhost"


def find_available_socket(
    custom_host: str | None = None,
    custom_range: range | None = None,
    will_close_then_reopen_socket: bool = False,
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
            return _bind_socket(
                host=current_host,
                port=port,
                will_close_then_reopen_socket=will_close_then_reopen_socket,
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
        # comment on find_available_socket()
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
        ) from err_info
    except OSError as err_info:
        if err_info.errno in {48, 98}:
            raise PortAlreadyInUseException(
                f"Port {port} already in use."
            ) from err_info
        raise OSError(f"Unknown `OSError` while binding port {port}") from err_info


def get_family(host: str) -> socket.AddressFamily:
    try:
        socket.inet_pton(socket.AF_INET6, host)
        return socket.AF_INET6
    except OSError:
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
    except BaseException:
        logger.warning("Cannot determine ip-address. Fallback to localhost...")
        retval = "127.0.0.1"
    finally:
        s.close()
    logger.debug(f"ip-address: {retval}")
    return retval
