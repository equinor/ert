from __future__ import annotations

import json
import logging
import os
import sys
import threading
from collections.abc import Mapping
from inspect import Traceback
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Any, cast

import requests

from ert.dark_storage.client import Client, ErtClientConnectionInfo
from ert.services._base_service import ErtServerConnectionInfo, _Proc
from ert.trace import get_traceparent


class ErtServerContext:
    def __init__(self, service: ErtServer) -> None:
        self._service = service

    def __enter__(self) -> ErtServer:
        return self._service

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: Traceback,
    ) -> bool:
        self._service.shutdown()
        return exc_type is None


class ErtServer:
    _instance: ErtServer | None = None

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

        run_storage_main_cmd = [
            sys.executable,
            str(Path(__file__).parent / "_storage_main.py"),
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

        if self._connection_info is not None:
            if isinstance(connection_info, Mapping) and "urls" not in connection_info:
                raise KeyError("urls not found in conn_info")

            self._on_connection_info_received_event.set()
            self._thread_that_starts_server_process = None
        else:
            self._thread_that_starts_server_process = _Proc(
                service_name="storage",
                exec_args=run_storage_main_cmd,
                timeout=120,
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
        return ErtServerContext(service)

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
    ) -> ErtServer:
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
                if (path / "storage_server.json").exists():
                    with (path / "storage_server.json").open() as f:
                        return ErtServer(
                            storage_path=str(path),
                            connection_info=json.load(f),
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
            path = f"{self._storage_path}/storage_server.json"
        else:
            path = "storage_server.json"

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
