"""
This file contains a more generic version of "ert services", and
is scheduled for removal when WebvizErt is removed.
"""

from __future__ import annotations

import json
import os
import threading
import types
from collections.abc import Mapping, Sequence
from logging import Logger, getLogger
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar

from ert.services.ert_server import ErtServerConnectionInfo, _Proc

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound="BaseService")


class _Context(Generic[T]):
    def __init__(self, service: T) -> None:
        self._service = service

    def __enter__(self) -> T:
        return self._service

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool:
        self._service.shutdown()
        return exc_type is None


class BaseService:
    """
    BaseService provides a block-only-when-needed mechanism for starting and
    maintaining services as subprocesses.

    This is achieved by using a POSIX communication pipe, over which the service
    can communicate that it has started. The contents of the communication is
    also written to a file inside of the ERT storage directory.

    The service itself can implement the other side of the pipe as such::

        import os

        # ... perform initialisation ...

        # BaseService provides this environment variable with the pipe's FD
        comm_fd = os.environ["ERT_COMM_FD"]

        # Open the pipe with Python's IO classes for ease of use
        with os.fdopen(comm_fd, "wb") as comm:
            # Write JSON over the pipe, which will be interpreted by a subclass
            # of BaseService on ERT's side
            comm.write('{"some": "json"}')

        # The pipe is flushed and closed here. This tells BaseService that
        # initialisation is finished and it will try to read the JSON data.
    """

    _instance: BaseService | None = None

    def __init__(
        self,
        exec_args: Sequence[str] = (),
        timeout: int = 120,
        conn_info: ErtServerConnectionInfo | Exception | None = None,
        project: str | None = None,
    ) -> None:
        self._exec_args = exec_args
        self._timeout = timeout

        self._proc: _Proc | None = None
        self._conn_info: ErtServerConnectionInfo | Exception | None = conn_info
        self._conn_info_event = threading.Event()
        self._project = Path(project) if project is not None else Path.cwd()

        # Flag that we have connection information
        if self._conn_info:
            self._conn_info_event.set()
        else:
            self._proc = _Proc(
                self.service_name, exec_args, timeout, self.set_conn_info, self._project
            )

    @classmethod
    def start_server(cls, *args: Any, **kwargs: Any) -> _Context[Self]:
        if cls._instance is not None:
            raise RuntimeError("Server already running")
        cls._instance = obj = cls(*args, **kwargs)
        if obj._proc is not None:
            obj._proc.start()
        return _Context(obj)

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
        name = f"{cls.service_name}_server.json"
        # Note: If the caller actually pass None, we override that here...
        if timeout is None:
            timeout = 240
        t = -1
        while t < timeout:
            if (path / name).exists():
                with (path / name).open() as f:
                    return cls((), conn_info=json.load(f), project=str(path))

            sleep(1)
            t += 1

        raise TimeoutError("Server not started")

    def wait_until_ready(self, timeout: int | None = None) -> bool:
        if timeout is None:
            timeout = self._timeout

        if self._conn_info_event.wait(timeout):
            return not (
                self._conn_info is None or isinstance(self._conn_info, Exception)
            )
        if isinstance(self._conn_info, TimeoutError):
            self.logger.critical(f"startup exceeded defined timeout {timeout}s")
        return False  # Timeout reached

    def wait(self) -> None:
        if self._proc is not None:
            self._proc.join()

    def set_conn_info(self, info: ErtServerConnectionInfo | Exception | None) -> None:
        if self._conn_info is not None:
            raise ValueError("Connection information already set")
        if info is None:
            raise ValueError
        self._conn_info = info

        if self._project is not None:
            if not Path(self._project).exists():
                raise RuntimeError(f"No storage exists at : {self._project}")
            path = f"{self._project}/{self.service_name}_server.json"
        else:
            path = f"{self.service_name}_server.json"

        if isinstance(info, Mapping):
            with NamedTemporaryFile(dir=f"{self._project}", delete=False) as f:
                f.write(json.dumps(info, indent=4).encode("utf-8"))
                f.flush()
                os.rename(f.name, path)

        self._conn_info_event.set()

    def fetch_conn_info(self) -> Mapping[str, Any]:
        is_ready = self.wait_until_ready(self._timeout)
        if isinstance(self._conn_info, Exception):
            raise self._conn_info
        if not is_ready:
            raise TimeoutError
        if self._conn_info is None:
            raise ValueError("conn_info is None")
        return self._conn_info

    def shutdown(self) -> int:
        """Shutdown the server."""
        if self._proc is None:
            return -1
        self.__class__._instance = None
        proc, self._proc = self._proc, None
        return proc.shutdown()

    @property
    def service_name(self) -> str:
        """
        Subclass should return the name of the service, eg 'storage' for ERT Storage.
        Used for identifying the server information JSON file.
        """
        raise NotImplementedError

    @property
    def logger(self) -> Logger:
        return getLogger(f"ert.shared.{self.service_name}")

    @property
    def _service_file(self) -> str:
        return f"{self.service_name}_server.json"
