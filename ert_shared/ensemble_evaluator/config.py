import os
import yaml
import logging

logger = logging.getLogger(__name__)
CLIENT_URI = "client"
DISPATCH_URI = "dispatch"
DEFAULT_PORT = "8765"
DEFAULT_HOST = "localhost"
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
