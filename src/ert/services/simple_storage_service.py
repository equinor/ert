from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from json import JSONDecodeError
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Self

from ert.trace import get_traceparent

from ..dark_storage.client import Client
from ._base_service import ConnInfo, _Context, _Proc


class ErtServer:
    _instance: ErtServer | None = None

    def __init__(self, storage_path: str, traceparent: str):
        self._storage_path = storage_path
        run_storage_main_cmd = [
            sys.executable,
            str(Path(__file__).parent / "storage_main.py"),
            "--project",
            storage_path,
            (get_traceparent() if traceparent == "inherit_parent" else traceparent),
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
    def start_server(cls, project: Path) -> _Context[T]:
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
