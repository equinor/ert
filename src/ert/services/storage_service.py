from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from json.decoder import JSONDecodeError
from typing import Any

import requests

from ert.dark_storage.client import Client, ConnInfo
from ert.services._base_service import BaseService, _Context, local_exec_args
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
        traceparent: str | None = "inherit_parent",
        logging_config: str | None = None,
    ) -> None:
        self._url: str | None = None

        exec_args = local_exec_args("storage")

        exec_args.extend(["--project", str(project)])
        if verbose:
            exec_args.append("--verbose")
        if logging_config:
            exec_args.extend(["--logging-config", str(logging_config)])
        if traceparent:
            traceparent = (
                get_traceparent() if traceparent == "inherit_parent" else traceparent
            )
            exec_args.extend(["--traceparent", str(traceparent)])
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
                if resp.status_code == 200:
                    self._url = url
                    return str(url)
                logging.getLogger(__name__).info(
                    f"Connecting to {url} got status: "
                    f"{resp.status_code}, {resp.headers}, {resp.reason}, {resp.text}"
                )

            except requests.ConnectionError as ce:
                logging.getLogger(__name__).info(f"Could not connect to {url}: {ce}")
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
