import yaml
import logging
import socket
from ert_shared.storage.main import bind_socket

logger = logging.getLogger(__name__)


def _get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


def find_open_port(lower=51820, upper=51840):
    host = _get_ip_address()
    for port in range(lower, upper):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, port))
            sock.close()
            return port
        except socket.error:
            pass
    msg = f"No open port for host {host} in the range {lower}-{upper}"
    logging.exception(msg)
    raise Exception(msg)


class EvaluatorServerConfig:
    def __init__(self, port=None):
        self.host = _get_ip_address()
        self.port = find_open_port() if port is None else port
        self.socket = bind_socket(self.host, self.port)
        self.url = f"ws://{self.host}:{self.port}"
        self.client_uri = f"{self.url}/client"
        self.dispatch_uri = f"{self.url}/dispatch"

    def get_socket(self):
        if self.socket._closed:
            self.socket = bind_socket(self.host, self.port)
            return self.socket
        return self.socket
