import os
import socket

from ert.shared import __version__


def find_ert_info() -> str:
    is_on_cluster = "KOMODO_RELEASE" in os.environ
    if is_on_cluster:
        hostname = socket.gethostname()
        return f"{os.environ['KOMODO_RELEASE']} @ {hostname}"
    return f"version {__version__}"
