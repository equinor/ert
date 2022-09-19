import logging
from os import PathLike
from typing import Any, Optional, Tuple

import httpx
import requests
from ert_storage.client import Client, ConnInfo

from ert.services._base_service import BaseService, _Context, local_exec_args


class Storage(BaseService):
    service_name = "storage"

    def __init__(
        self,
        res_config: Optional[PathLike] = None,
        database_url: str = "sqlite:///ert.db",
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        self._url: Optional[str] = None

        exec_args = local_exec_args("storage")
        if res_config:
            exec_args.append(str(res_config))
        else:
            exec_args.extend(("--database-url", database_url))
        if verbose:
            exec_args.append("--verbose")
        if "project" in kwargs:
            exec_args.extend(["--project", str(kwargs["project"])])

        super().__init__(exec_args, *args, **kwargs)

    def fetch_auth(self) -> Tuple[str, Any]:
        """
        Returns a tuple of username and password, compatible with requests' `auth`
        kwarg.

        Blocks while the server is starting.
        """
        return ("__token__", self.fetch_conn_info()["authtoken"])

    @classmethod
    def init_service(cls, *args: Any, **kwargs: Any) -> _Context:
        try:
            service = cls.connect(timeout=0, project=kwargs.get("project"))
            # Check the server is up and running
            _ = service.fetch_url()
        except TimeoutError:
            return cls.start_server(*args, **kwargs)
        return _Context(service)

    def fetch_url(self) -> str:
        """Returns the url. Blocks while the server is starting"""
        if self._url is not None:
            return self._url

        for url in self.fetch_conn_info()["urls"]:
            try:
                resp = requests.get(f"{url}/healthcheck", auth=self.fetch_auth())
                if resp.status_code == 200:
                    self._url = url
                    return url
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
    def session(cls, timeout: Optional[int] = None) -> Client:
        """
        Start a HTTP transaction with the server
        """
        inst = cls.connect(timeout=timeout)
        return Client(
            conn_info=ConnInfo(
                base_url=inst.fetch_url(), auth_token=inst.fetch_auth()[1]
            )
        )

    @classmethod
    async def async_session(cls, timeout: Optional[int] = None) -> httpx.AsyncClient:
        """
        Start a HTTP transaction with the server
        """
        inst = cls.connect(timeout=timeout)
        base_url = inst.fetch_url()
        token = inst.fetch_auth()[1]
        return httpx.AsyncClient(base_url=base_url, headers={"Token": token})
