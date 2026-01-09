from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import signal
import sys
import threading
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from select import PIPE_BUF, select
from subprocess import Popen, TimeoutExpired
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Any, TypedDict, cast

import requests

from ert.dark_storage.client import Client, ErtClientConnectionInfo
from ert.trace import get_traceparent

SERVICE_CONF_PATHS: set[str] = set()


class ErtServerConnectionInfo(TypedDict):
    urls: list[str]
    authtoken: str
    host: str
    port: str
    cert: str
    auth: str


class BaseServiceExit(OSError):
    pass


def cleanup_service_files(signum: int, frame: types.FrameType | None) -> None:
    for file_path in SERVICE_CONF_PATHS:
        file = Path(file_path)
        if file.exists():
            file.unlink()
    raise BaseServiceExit(f"Signal {signum} received.")


if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGTERM, cleanup_service_files)
    signal.signal(signal.SIGINT, cleanup_service_files)


class ServerBootFail(RuntimeError):
    pass


class _Proc(threading.Thread):
    def __init__(
        self,
        service_name: str,
        exec_args: Sequence[str],
        timeout: int,
        on_connection_info_received: Callable[
            [ErtServerConnectionInfo | Exception | None], None
        ],
        project: Path,
    ) -> None:
        super().__init__()

        self._shutdown = threading.Event()

        self._service_name = service_name
        self._exec_args = exec_args
        self._timeout = timeout
        self._propagate_connection_info_from_childproc = on_connection_info_received
        self._service_config_path = project / f"{self._service_name}_server.json"

        fd_read, fd_write = os.pipe()
        self._comm_pipe = os.fdopen(fd_read)

        env = os.environ.copy()
        env["ERT_COMM_FD"] = str(fd_write)

        SERVICE_CONF_PATHS.add(str(self._service_config_path))

        # The process is waited for in _do_shutdown()
        self._childproc = Popen(
            self._exec_args,
            pass_fds=(fd_write,),
            env=env,
            close_fds=True,
        )
        os.close(fd_write)

    def run(self) -> None:
        comm = self._read_connection_info_from_process(self._childproc)

        if comm is None:
            self._propagate_connection_info_from_childproc(TimeoutError())
            return  # _read_conn_info() has already cleaned up in this case

        conn_info: ErtServerConnectionInfo | Exception | None = None
        try:
            conn_info = json.loads(comm)
        except json.JSONDecodeError:
            conn_info = ServerBootFail()
        except Exception as exc:
            conn_info = exc

        try:
            self._propagate_connection_info_from_childproc(conn_info)

            while True:
                if self._childproc.poll() is not None:
                    break
                if self._shutdown.wait(1):
                    self._do_shutdown()
                    break

        except Exception as e:
            print(str(e))
            self.logger.exception(e)

        finally:
            self._ensure_connection_info_file_is_deleted()

    def shutdown(self) -> int:
        """Shutdown the server."""
        self._shutdown.set()
        self.join()

        return self._childproc.returncode

    def _read_connection_info_from_process(self, proc: Popen[bytes]) -> str | None:
        comm_buf = io.StringIO()
        first_iter = True
        while first_iter or proc.poll() is None:
            first_iter = False
            ready = select([self._comm_pipe], [], [], self._timeout)

            # Timeout reached, exit with a failure
            if ready == ([], [], []):
                self._do_shutdown()
                self._ensure_connection_info_file_is_deleted()
                return None

            x = self._comm_pipe.read(PIPE_BUF)
            if not x:  # EOF
                break
            comm_buf.write(x)
        return comm_buf.getvalue()

    def _do_shutdown(self) -> None:
        if self._childproc is None:
            return
        try:
            self._childproc.terminate()
            self._childproc.wait(10)  # Give it 10s to shut down cleanly..
        except TimeoutExpired:
            try:
                self._childproc.kill()  # ... then kick it harder...
                self._childproc.wait(self._timeout)  # ... and wait again
            except TimeoutExpired:
                self.logger.error(
                    f"waiting for child-process exceeded timeout {self._timeout}s"
                )

    def _ensure_connection_info_file_is_deleted(self) -> None:
        """
        Ensure that the JSON connection information file is deleted
        """
        with contextlib.suppress(OSError):
            if self._service_config_path.exists():
                self._service_config_path.unlink()

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger("ert.shared.storage")


_ERT_SERVER_CONNECTION_INFO_FILE = "storage_server.json"
_ERT_SERVER_EXECUTABLE_FILE = str(Path(__file__).parent / "_storage_main.py")


class ErtServerContext:
    def __init__(self, service: ErtServerConnection) -> None:
        self._service = service

    def __enter__(self) -> ErtServerConnection:
        return self._service

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool:
        self._service.shutdown()
        return exc_type is None


class ErtServerConnection:
    _instance: ErtServerConnection | None = None

    def __init__(
        self,
        storage_path: str,
        timeout: int = 120,
        parent_pid: int | None = None,
        connection_info: ErtServerConnectionInfo | Exception | None = None,
        verbose: bool = False,
        logging_config: str | None = None,  # Only used from everserver
    ) -> None:
        if timeout is None:
            timeout = 120

        self._storage_path = storage_path
        self._connection_info: ErtServerConnectionInfo | Exception | None = (
            connection_info
        )
        self._on_connection_info_received_event = threading.Event()
        self._timeout = timeout
        self._url: str | None = None

        if self._connection_info is not None:
            # This means that server is already running
            if isinstance(connection_info, Mapping) and "urls" not in connection_info:
                raise KeyError("No URLs found in connection info")

            self._on_connection_info_received_event.set()
            self._thread_that_starts_server_process = None
            return

        run_storage_main_cmd = [
            sys.executable,
            _ERT_SERVER_EXECUTABLE_FILE,
            "--project",
            storage_path,
        ]

        if logging_config is not None:
            run_storage_main_cmd += ["--logging-config", logging_config]

            traceparent = get_traceparent()
            if traceparent is not None:
                run_storage_main_cmd += ["--traceparent", traceparent]

        if parent_pid is not None:
            run_storage_main_cmd += ["--parent_pid", str(parent_pid)]

        if verbose:
            run_storage_main_cmd.append("--verbose")

        self._thread_that_starts_server_process = _Proc(
            service_name="storage",
            exec_args=run_storage_main_cmd,
            timeout=timeout,
            on_connection_info_received=self.on_connection_info_received_from_server_process,
            project=Path(self._storage_path),
        )

    def fetch_auth(self) -> tuple[str, Any]:
        """
        Returns a tuple of username and password, compatible with requests' `auth`
        kwarg.

        Blocks while the server is starting.
        """
        return (
            "__token__",
            cast(dict[str, Any], self.fetch_connection_info())["authtoken"],
        )

    @classmethod
    def init_service(
        cls,
        project: Path,
        timeout: int = 0,
        logging_config: str | None = None,
    ) -> ErtServerContext:
        try:
            service = cls.connect(
                project=project or Path.cwd(), timeout=0, logging_config=logging_config
            )
            # Check the server is up and running
            _ = service.fetch_url()
            return ErtServerContext(service)
        except (TimeoutError, json.JSONDecodeError, KeyError) as e:
            logging.getLogger(__name__).warning(
                "Failed locating existing storage service due to "
                f"{type(e).__name__}: {e}, starting new service"
            )
            return cls.start_server(
                project=project, timeout=timeout, logging_config=logging_config
            )
        except PermissionError as pe:
            logging.getLogger(__name__).error(
                f"{type(pe).__name__}: {pe}, cannot connect to storage service "
                f"due to permission issues."
            )
            raise pe

    def fetch_url(self) -> str:
        """Returns the url. Blocks while the server is starting"""
        if self._url is not None:
            return self._url

        for url in self.fetch_connection_info()["urls"]:
            con_info = self.fetch_connection_info()
            try:
                resp = requests.get(
                    f"{url}/healthcheck",
                    auth=self.fetch_auth(),
                    verify=con_info["cert"],
                )
                logging.getLogger(__name__).info(
                    f"Connecting to {url} got status: "
                    f"{resp.status_code}, {resp.headers}, {resp.reason}, {resp.text}"
                )
                if resp.status_code == 200:
                    self._url = url
                    return str(url)

            except requests.ConnectionError as ce:
                logging.getLogger(__name__).info(
                    f"Could not connect to {url}, but will try something else. "
                    f"Error: {ce}"
                )
        raise TimeoutError(
            "None of the URLs provided for the ert storage server worked."
        )

    @classmethod
    def session(cls, project: os.PathLike[str], timeout: int | None = None) -> Client:
        """
        Start a HTTP transaction with the server
        """
        inst = cls.connect(timeout=timeout, project=project)
        info = inst.fetch_connection_info()
        return Client(
            conn_info=ErtClientConnectionInfo(
                base_url=inst.fetch_url(),
                auth_token=inst.fetch_auth()[1],
                cert=info["cert"],
            )
        )

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger("ert.shared.storage")

    def shutdown(self) -> int:
        """Shutdown the server."""
        if self._thread_that_starts_server_process is None:
            return -1

        self.__class__._instance = None
        error_code = self._thread_that_starts_server_process.shutdown()
        self._thread_that_starts_server_process = None

        return error_code

    @classmethod
    def connect(
        cls,
        *,
        project: os.PathLike[str],
        timeout: int | None = None,
        logging_config: str | None = None,
    ) -> ErtServerConnection:
        if cls._instance is not None:
            cls._instance.wait_until_ready()
            assert isinstance(cls._instance, cls)
            return cls._instance

        path = Path(project)

        # Wait for storage_server.json file to appear
        try:
            if timeout is None:
                timeout = 240
            t = -1
            while t < timeout:
                storage_server_path = path / _ERT_SERVER_CONNECTION_INFO_FILE
                if (
                    storage_server_path.exists()
                    and storage_server_path.stat().st_size > 0
                ):
                    with (path / _ERT_SERVER_CONNECTION_INFO_FILE).open() as f:
                        storage_server_content = json.load(f)

                    return ErtServerConnection(
                        storage_path=str(path),
                        connection_info=storage_server_content,
                        logging_config=logging_config,
                    )

                sleep(1)
                t += 1

            raise TimeoutError("Server not started")
        except PermissionError as pe:
            logging.getLogger(__name__).error(
                f"{type(pe).__name__}: {pe}, cannot connect to ert server service "
                f"due to permission issues."
            )
            raise pe

    @classmethod
    def start_server(
        cls,
        project: Path,
        parent_pid: int | None = None,
        verbose: bool = False,
        timeout: int | None = None,
        logging_config: str | None = None,
    ) -> ErtServerContext:
        if cls._instance is not None:
            raise RuntimeError("Server already running")
        cls._instance = obj = cls(
            storage_path=str(project),
            parent_pid=parent_pid,
            verbose=verbose,
            timeout=timeout or 120,
            logging_config=logging_config,
        )
        if obj._thread_that_starts_server_process is not None:
            obj._thread_that_starts_server_process.start()
        return ErtServerContext(obj)

    def on_connection_info_received_from_server_process(
        self, info: ErtServerConnectionInfo | Exception | None
    ) -> None:
        if self._connection_info is not None:
            raise ValueError("Connection information already set")
        if info is None:
            raise ValueError
        self._connection_info = info

        if self._storage_path is not None:
            if not Path(self._storage_path).exists():
                raise RuntimeError(f"No storage exists at : {self._storage_path}")
            path = f"{self._storage_path}/{_ERT_SERVER_CONNECTION_INFO_FILE}"
        else:
            path = _ERT_SERVER_CONNECTION_INFO_FILE

        if isinstance(info, Mapping):
            with NamedTemporaryFile(dir=f"{self._storage_path}", delete=False) as f:
                f.write(json.dumps(info, indent=4).encode("utf-8"))
                f.flush()
                os.rename(f.name, path)

        self._on_connection_info_received_event.set()

    def wait_until_ready(self, timeout: int | None = None) -> bool:
        if timeout is None:
            timeout = self._timeout

        if self._on_connection_info_received_event.wait(timeout):
            return not (
                self._connection_info is None
                or isinstance(self._connection_info, Exception)
            )
        if isinstance(self._connection_info, TimeoutError):
            self.logger.critical(f"startup exceeded defined timeout {timeout}s")
        return False  # Timeout reached

    def fetch_connection_info(self) -> ErtServerConnectionInfo:
        is_ready = self.wait_until_ready(self._timeout)
        if isinstance(self._connection_info, Exception):
            raise self._connection_info
        if not is_ready:
            raise TimeoutError
        if self._connection_info is None:
            raise ValueError("conn_info is None")
        return self._connection_info

    def wait(self) -> None:
        if self._thread_that_starts_server_process is not None:
            self._thread_that_starts_server_process.join()
