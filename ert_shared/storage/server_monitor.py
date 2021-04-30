import io
import os
import sys
import threading
import json
import requests
from asyncio import TimeoutError
from select import select, PIPE_BUF
from subprocess import Popen, TimeoutExpired
from pathlib import Path
from typing import Any, Tuple


def empty(arr):
    return len(arr) == 0


class ServerBootFail(RuntimeError):
    pass


class ServerMonitor(threading.Thread):
    EXEC_ARGS = [sys.executable, "-m", "ert_shared.storage"]
    TIMEOUT = 20  # Wait 20s for the server to start before panicking
    _instance = None

    def __init__(self, *, rdb_url=None, lockfile=True):
        super().__init__()

        self._assert_server_not_running()

        self._url = None
        self._connection_info = None

        self._server_info_lock = threading.Lock()
        self._server_info_lock.acquire()

        env = os.environ.copy()
        args = []
        if not lockfile:
            args.append("--disable-lockfile")
        if rdb_url:
            args.extend(("--rdb-url", rdb_url))
            env["ERT_STORAGE_DB"] = str(rdb_url)

        fd_read, fd_write = os.pipe()
        self._comm_pipe = os.fdopen(fd_read)

        env["ERT_COMM_FD"] = str(fd_write)
        self._proc = Popen(
            self.EXEC_ARGS + args,
            pass_fds=(fd_write,),
            env=env,
            close_fds=True,
            stdout=open("storage_access.log", "w"),
        )
        os.close(fd_write)

    def run(self):
        comm_buf = io.StringIO()
        while self._proc.poll() is None:
            ready = select([self._comm_pipe], [], [], self.TIMEOUT)

            # Timeout reached, exit with a failure
            if all(map(empty, ready)):
                self.shutdown()
                self._server_info_lock.release()
                self._connection_info = TimeoutError()
                return

            x = self._comm_pipe.read(PIPE_BUF)
            if x == "":  # EOF
                break
            comm_buf.write(x)

        try:
            self._connection_info = json.loads(comm_buf.getvalue())
        except json.JSONDecodeError:
            self._connection_info = ServerBootFail()
        except Exception as e:
            self._connection_info = e

        self._server_info_lock.release()
        self._proc.wait()

    def fetch_auth(self) -> Tuple[str, Any]:
        """Returns a tuple of username and password, compatible with requests' `auth`
        kwarg.

        Blocks while the server is starting.

        """
        return ("__token__", self.fetch_connection_info()["authtoken"])

    def fetch_url(self) -> str:
        """Returns the url. Blocks while the server is starting"""
        if self._url is not None:
            return self._url

        for url in self.fetch_connection_info()["urls"]:
            try:
                resp = requests.get(f"{url}/healthcheck", auth=self.fetch_auth())
                if resp.status_code == 200:
                    self._url = url
                    return url
            except requests.ConnectionError:
                pass

        raise RuntimeError("Server started, but none of the URLs provided worked")

    def fetch_connection_info(self):
        """Retrieves the authnetication token. Blocks while the server is starting."""
        with self._server_info_lock:
            info = self._connection_info
            if isinstance(info, Exception):
                raise info
            else:
                return info

    def shutdown(self):
        """Shutdown the server."""
        try:
            self._proc.terminate()
            return self._proc.wait(10)
        except TimeoutExpired:
            self._proc.kill()
        return self._proc.wait()

    def _assert_server_not_running(self):
        """It doesn't seem to be possible to check whether a server has been started
        other than looking for files that were created during the startup process.

        """
        if (Path.cwd() / "storage_server.json").exists():
            print(
                "A file called storage_server.json is present from this location. "
                "This indicates there is already a ert instance running. If you are "
                "certain that is not the case, try to delete the file and try "
                "again."
            )
            sys.exit(1)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
