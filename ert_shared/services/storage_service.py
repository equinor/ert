from os import PathLike
import sys
import httpx
import requests
from urllib.parse import urljoin
from pathlib import Path
from typing import Any, Optional, Mapping, Tuple
from ert_shared.services._base_service import BaseService, local_exec_args


class _Session(requests.Session):
    def __init__(self, base_url: str, headers: Mapping[str, str]):
        super().__init__()
        self._headers = headers
        self._base_url = base_url

    def request(  # type: ignore[override]
        self, method: str, url: str, *args: Any, **kwargs: Any
    ) -> requests.Response:
        kwargs.setdefault("headers", {})
        kwargs["headers"].update(self._headers)
        return super().request(method, urljoin(self._base_url, url), *args, **kwargs)


class Storage(BaseService):
    service_name = "storage"

    def __init__(
        self,
        res_config: Optional[PathLike] = None,
        database_url: str = "sqlite:///ert.db",
        *args,
        **kwargs,
    ):
        self._url: Optional[str] = None

        exec_args = local_exec_args("storage")
        if res_config:
            exec_args.append(str(res_config))
        else:
            exec_args.extend(("--database-url", database_url))

        super().__init__(exec_args, *args, **kwargs)

    def fetch_auth(self) -> Tuple[str, Any]:
        """
        Returns a tuple of username and password, compatible with requests' `auth`
        kwarg.

        Blocks while the server is starting.
        """
        return ("__token__", self.fetch_conn_info()["authtoken"])

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
            except requests.ConnectionError:
                pass

        raise RuntimeError("Server started, but none of the URLs provided worked")

    @classmethod
    def session(cls, timeout=None) -> requests.Session:
        """
        Start a HTTP transaction with the server
        """
        inst = cls.connect(timeout=timeout)
        base_url = inst.fetch_url()
        token = inst.fetch_auth()[1]
        return _Session(base_url=base_url, headers={"Token": token})

    @classmethod
    async def async_session(cls, timeout=None) -> httpx.AsyncClient:
        """
        Start a HTTP transaction with the server
        """
        inst = cls.connect(timeout=timeout)
        base_url = inst.fetch_url()
        token = inst.fetch_auth()[1]
        return httpx.AsyncClient(base_url=base_url, headers={"Token": token})
