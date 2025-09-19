import ipaddress
import logging
import random
import socket
from functools import lru_cache

import psutil
from dns import exception, resolver, reversename


class PortAlreadyInUseException(Exception):
    pass


class NoPortsInRangeException(Exception):
    pass


class InvalidHostException(Exception):
    pass


logger = logging.getLogger(__name__)


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
        ip_addr = socket.gethostbyname(hostname)
        reverse_name = reversename.from_address(ip_addr)
        resolved_hosts = [
            str(ptr_record).rstrip(".")
            for ptr_record in resolver.resolve(reverse_name, "PTR")
        ]
        resolved_hosts.sort()
        return resolved_hosts[0]
    except (resolver.NXDOMAIN, exception.Timeout, resolver.NoResolverConfiguration):
        # If local address and reverse lookup not working - fallback
        # to socket fqdn which are using /etc/hosts to retrieve this name
        return socket.getfqdn()
    except (socket.gaierror, exception.DNSException):
        return "localhost"


def find_available_socket(
    host: str | None = None,
    port_range: range = range(51820, 51840 + 1),
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

    try:
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


def get_ip_address() -> str:
    """
    Get the first private IPv4 address of the current machine on the LAN.
    Returns the private IP, public IP if no private is found,
    or loopback as a last resort.

    Returns:
        str: The selected IP address as a string.
    """
    loopback = ""
    public = ""
    interfaces = psutil.net_if_addrs()
    for addresses in interfaces.values():
        for address in addresses:
            if address.family.name == "AF_INET":
                ip = address.address
                if ipaddress.ip_address(ip).is_loopback:
                    loopback = ip
                elif ipaddress.ip_address(ip).is_private:
                    return ip
                else:
                    public = ip

    if public or loopback:
        return public or loopback  # returns first non-None value
    else:
        logger.warning("Cannot determine ip-address. Falling back to 127.0.0.1")
        return "127.0.0.1"
