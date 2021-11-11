import os
import requests
import json

from urllib.parse import urljoin
from typing import Any, Tuple, Optional


class Session(requests.Session):
    def __init__(self):
        super().__init__()
        self._base_url: Optional[str] = None
        self._headers: Optional[Tuple] = ()

        # Should the storage_server.json also be used?
        # If so is there order of prefence
        if "ERT_STORAGE_CONNECTION_STRING" in os.environ:
            try:
                self._connection_info = json.loads(
                    os.environ["ERT_STORAGE_CONNECTION_STRING"]
                )
            except json.JSONDecodeError:
                raise RuntimeError("Invalid Storage Connection configuration")

        if {"urls", "authtoken"} <= self._connection_info.keys():
            self._base_url = self.__resolve_url()
            self._headers = {"Token": self._connection_info["authtoken"]}
        else:
            raise RuntimeError("Invalid Storage Connection configuration")

    def __enter__(self):
        # Do health check?
        # Seems redudant
        return self

    def __exit__(self, *args):
        # No real cleanup todo?
        pass

    def __resolve_url(self) -> str:
        """Resolve which of the candidate base urls to use."""
        for url in self._connection_info["urls"]:
            try:
                # Orginal code has auth token passed but is it actually used?
                resp = requests.get(f"{url}/healthcheck")
                if resp.status_code == 200:
                    return url
            except requests.ConnectionError:
                pass
        # Needs better exception message
        raise RuntimeError("None of the URLs provided worked")

    def request(  # type: ignore[override]
        self, method: str, url: str, *args: Any, **kwargs: Any
    ) -> requests.Response:
        kwargs.setdefault("headers", {})
        kwargs["headers"].update(self._headers)
        return super().request(method, urljoin(self._base_url, url), *args, **kwargs)
