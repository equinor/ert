from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from json.decoder import JSONDecodeError
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Any, Self

import requests

from ert.dark_storage.client import Client, ConnInfo
from ert.services._base_service import BaseService, _Context, _Proc, local_exec_args
from ert.trace import get_traceparent


class StorageService(BaseService):
    service_name = "storage"

    def __init__(
        self,
        exec_args: Sequence[str] = (),
        timeout: int = 120,
        parent_pid: int | None = None,
        conn_info: Mapping[str, Any] | Exception | None = None,
        project: str | None = None,
        verbose: bool = False,
        logging_config: str | None = None,
    ) -> None:
        self._url: str | None = None

        exec_args = local_exec_args("storage")

        exec_args.extend(["--project", str(project)])
        if verbose:
            exec_args.append("--verbose")
        if logging_config:
            exec_args.extend(["--logging-config", str(logging_config)])

            traceparent = get_traceparent()
            if traceparent is not None:
                exec_args.extend(["--traceparent", traceparent])

        if parent_pid is not None:
            exec_args.extend(["--parent_pid", str(parent_pid)])

        if (
            conn_info is not None
            and isinstance(conn_info, Mapping)
            and "urls" not in conn_info
        ):
            raise KeyError("urls not found in conn_info")
        super().__init__(exec_args, timeout, conn_info, project)

    def fetch_auth(self) -> tuple[str, Any]:
        """
        Returns a tuple of username and password, compatible with requests' `auth`
        kwarg.

        Blocks while the server is starting.
        """
        return ("__token__", self.fetch_conn_info()["authtoken"])

    @classmethod
    def init_service(cls, *args: Any, **kwargs: Any) -> _Context[StorageService]:
        try:
            service = cls.connect(timeout=0, project=kwargs.get("project", os.getcwd()))
            # Check the server is up and running
            _ = service.fetch_url()
        except (TimeoutError, JSONDecodeError, KeyError) as e:
            logging.getLogger(__name__).warning(
                "Failed locating existing storage service due to "
                f"{type(e).__name__}: {e}, starting new service"
            )
            return cls.start_server(*args, **kwargs)
        except PermissionError as pe:
            logging.getLogger(__name__).error(
                f"{type(pe).__name__}: {pe}, cannot connect to storage service "
                f"due to permission issues."
            )
            raise pe
        return _Context(service)

    def fetch_url(self) -> str:
        """Returns the url. Blocks while the server is starting"""
        if self._url is not None:
            return self._url

        for url in self.fetch_conn_info()["urls"]:
            con_info = self.fetch_conn_info()
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
        info = inst.fetch_conn_info()
        return Client(
            conn_info=ConnInfo(
                base_url=inst.fetch_url(),
                auth_token=inst.fetch_auth()[1],
                cert=info["cert"],
            )
        )


class ErtServer:
    _instance: ErtServer | None = None

    def __init__(self, storage_path: str):
        self._storage_path = storage_path
        run_storage_main_cmd = [
            sys.executable,
            str(Path(__file__).parent / "storage_main.py"),
            "--project",
            storage_path,
            get_traceparent(),
        ]
        self._thread_around_server_process = _Proc(
            service_name="storage",
            exec_args=run_storage_main_cmd,
            timeout=120,
            set_conn_info=self.set_conn_info,
        )
        self._thread_around_server_process.run()

    def shutdown(self):
        pass

    @classmethod
    def connect(
        cls,
        *,
        project: os.PathLike[str],
        timeout: int | None = None,
    ) -> Self:
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
                        cls._instance = ErtServer(project=str(path))
                        cls._instance.wait_until_ready()
                        return cls._instance

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
    def start_server(cls, project: Path) -> _Context[ErtServer]:
        if cls._instance is not None:
            raise RuntimeError("Server already running")
        cls._instance = obj = cls(project)
        if obj._proc is not None:
            obj._proc.start()
        return _Context(obj)

    @classmethod
    def init_service(cls, project: Path) -> _Context[ErtServer]:
        try:
            service = cls.connect(timeout=0, project=project or os.getcwd())
            # Check the server is up and running
            _ = service.fetch_url()
        except (TimeoutError, JSONDecodeError, KeyError) as e:
            logging.getLogger(__name__).warning(
                "Failed locating existing storage service due to "
                f"{type(e).__name__}: {e}, starting new service"
            )
            return cls.start_server(project)
        except PermissionError as pe:
            logging.getLogger(__name__).error(
                f"{type(pe).__name__}: {pe}, cannot connect to storage service "
                f"due to permission issues."
            )
            raise pe
        return _Context(service)

    def set_conn_info(self, info: ConnInfo) -> None:
        if self._conn_info is not None:
            raise ValueError("Connection information already set")
        if info is None:
            raise ValueError
        self._conn_info = info

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

        self._conn_info_event.set()

    @classmethod
    def session(cls, project: os.PathLike[str], timeout: int | None = None) -> Client:
        """
        Start a HTTP transaction with the server
        """
        inst = cls.connect(timeout=timeout, project=project)
        info = inst.fetch_conn_info()
        return Client(
            conn_info=ConnInfo(
                base_url=inst.fetch_url(),
                auth_token=inst.fetch_auth()[1],
                cert=info["cert"],
            )
        )


# StorageService = ErtServer # tmpy tmpy
