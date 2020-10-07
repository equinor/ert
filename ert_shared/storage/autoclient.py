import logging

logger = logging.getLogger(__name__)
import os
import socket
import subprocess
import sys

import requests

from ert_shared.storage.client import StorageClient


class AutoClient(StorageClient):
    """A client that will start its own server, and tears it down on request."""

    def __init__(self, bind):
        StorageClient.__init__(self, "")

        (bind_host, bind_port) = bind.split(":")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((bind_host, int(bind_port)))
        sock.listen()
        address, port = sock.getsockname()

        self._BASE_URI = "{}://{}:{}".format("http", address, port)

        logger.info("Serving Storage API on {}".format(self._BASE_URI))

        # XXX: a hack to get flask to pick up our created socket, for more
        # serious application servers, this would be passed explicitly to the
        # server interface.
        sock_fd = str(sock.fileno())
        os.environ["WERKZEUG_SERVER_FD"] = sock_fd

        bind = "{}:{}".format(address, port)
        self._server_proc = subprocess.Popen(
            [sys.argv[0], "api", "--bind", bind], pass_fds=(sock_fd,)
        )

    def shutdown(self):
        """Shut down the server that AutoClient created."""
        requests.post("{}/shutdown".format(self._BASE_URI))
        self._server_proc.wait(timeout=10)


# _Namespace allows the AutoClient to pass an argparse Namespace to run_server.
class _Namespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
