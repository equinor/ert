import yaml
import logging
import socket

logger = logging.getLogger(__name__)


def _get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


def find_open_port(lower, upper):
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


CLIENT_URI = "client"
DISPATCH_URI = "dispatch"
DEFAULT_PORT = find_open_port(lower=51820, upper=51840)
DEFAULT_HOST = _get_ip_address()
DEFAULT_URL = f"ws://{DEFAULT_HOST}:{DEFAULT_PORT}"
CONFIG_FILE = "ee_config.yml"

DEFAULT_EE_CONFIG = {
    "host": DEFAULT_HOST,
    "port": DEFAULT_PORT,
    "url": DEFAULT_URL,
    "client_url": f"{DEFAULT_URL}/{CLIENT_URI}",
    "dispatch_url": f"{DEFAULT_URL}/{DISPATCH_URI}",
}


def load_config(config_path=None):
    if config_path is None:
        return DEFAULT_EE_CONFIG.copy()

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
        host = data.get("host", DEFAULT_HOST)
        port = data.get("port", DEFAULT_PORT)
        return {
            "host": host,
            "port": port,
            "url": f"ws://{host}:{port}",
            "client_url": f"ws://{host}:{port}/{CLIENT_URI}",
            "dispatch_url": f"ws://{host}:{port}/{DISPATCH_URI}",
        }
