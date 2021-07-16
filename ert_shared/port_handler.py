import random
import socket
from typing import Optional, Tuple


class PortAlreadyInUseException(Exception):
    pass


class NoPortsInRangeException(Exception):
    pass


class InvalidHostException(Exception):
    pass


def find_available_port(
    custom_host: Optional[str] = None,
    custom_range: Optional[range] = None,
) -> Tuple[str, int]:
    current_host = custom_host if custom_host is not None else _get_ip_address()
    current_range = (
        custom_range if custom_range is not None else range(51820, 51840 + 1)
    )

    if current_range.start == current_range.stop:
        return current_host, current_range.start

    attempts = 0
    while attempts < current_range.stop - current_range.start:
        try:
            attempts += 1
            num = random.randrange(current_range.start, current_range.stop)
            sock = _bind_socket(host=current_host, port=num)
            sock.close()
            return current_host, num
        except PortAlreadyInUseException:
            continue

    raise NoPortsInRangeException(f"No available ports in predefined {current_range}.")


def get_socket(host: str, port: int) -> socket.socket:
    return _bind_socket(host=host, port=port, reuse_addr=True)


def _bind_socket(host: str, port: int, reuse_addr: bool = False) -> socket.socket:
    try:
        family = get_family(host=host)
        sock = socket.socket(family=family, type=socket.SOCK_STREAM)
        if reuse_addr:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
        if err_info.errno == 48 or err_info.errno == 98:
            raise PortAlreadyInUseException(f"Port {port} already in use.")
        raise Exception(
            f"Unknown `OSError` while binding port {port}. Actual "
            f"error msg is: {err_info.strerror}"
        )


def get_family(host: str) -> socket.AddressFamily:
    try:
        socket.inet_pton(socket.AF_INET6, host)
        return socket.AF_INET6
    except socket.error:
        return socket.AF_INET


def _get_ip_address() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]
