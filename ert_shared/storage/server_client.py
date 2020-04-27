import os
import subprocess
import sys

import requests

from ert_shared.storage.client import StorageClient


class ServerClient(StorageClient):
    """A client that will start its own non-development server, and tear it 
    down on request.
    """

    def __init__(self, bind):
        StorageClient.__init__(self, "", bind=bind)

        sock_fd = str(self._sock.fileno())
        bind = "fd://{}".format(sock_fd)
        self._server_proc = subprocess.Popen(
            [sys.argv[0], "api", "--bind", bind, "--production"], pass_fds=(sock_fd,)
        )

    def shutdown(self):
        requests.post("{}/shutdown".format(self._BASE_URI))
        # gunicorn has 30 sec graceful shutdown timeout
        self._server_proc.wait(timeout=31)


class DevServerClient(StorageClient):
    """A client that will start its own dev server, and tear it down on request.
    """

    def __init__(self, bind):
        StorageClient.__init__(self, "", bind=bind)

        # XXX: a hack to get flask to pick up our created socket
        sock_fd = str(self._sock.fileno())
        os.environ["WERKZEUG_SERVER_FD"] = sock_fd

        address, port = self._sock.getsockname()
        bind = "{}:{}".format(address, port)
        self._server_proc = subprocess.Popen(
            [sys.argv[0], "api", "--bind", bind], pass_fds=(sock_fd,)
        )

    def shutdown(self):
        """Shut down the server that DevServerClient created."""
        requests.post("{}/shutdown".format(self._BASE_URI))
        self._server_proc.wait(timeout=10)
