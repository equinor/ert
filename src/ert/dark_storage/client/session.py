import os
import requests
import json
from pathlib import Path
from urllib.parse import urljoin
from typing import Any, Dict, Type
from types import TracebackType


class Session(requests.Session):
    """Wrapper class for requests.Session to configure base url and auth token.

    The base url and auth token are read from either:
    The file storage_server.json, in the current working directory or
    The environment variable ERT_STORAGE_CONNECTION_STRING

    In both cases the configuration is represeted by JSON:

    {
     "urls": [list of base urls]
     "authtoken": auth_token
    }

    The list of base urls are tested via the /healthcheck api,
    with the first url returning a 200 status code is used.

    If both file and environment variable exist the environment variable is used.
    """

    def __init__(self) -> None:
        super().__init__()
        self._base_url: str = ""
        self._headers: Dict[str, str] = {}
        self._connection_info: Dict[str, Any] = {}

        connection_string = None

        connection_config = Path.cwd() / "storage_server.json"
        if connection_config.exists():
            with open(connection_config) as f:
                connection_string = f.read()

        # Allow env var to overide config file
        if "ERT_STORAGE_CONNECTION_STRING" in os.environ:
            connection_string = os.environ["ERT_STORAGE_CONNECTION_STRING"]

        if connection_string is None:
            raise RuntimeError("No Storage Connection configuration found")

        try:
            self._connection_info = json.loads(connection_string)
        except json.JSONDecodeError:
            raise RuntimeError("Invalid Storage Connection configuration")

        if {"urls", "authtoken"} <= self._connection_info.keys():
            self._base_url = self._resolve_url()
            self._headers = {"Token": self._connection_info["authtoken"]}
        else:
            raise RuntimeError("Invalid Storage Connection configuration")

    def __enter__(self) -> "Session":
        return self

    def __exit__(  # type: ignore[override]
        self,
        exc_value: BaseException,
        exc_type: Type[BaseException],
        traceback: TracebackType,
    ) -> bool:
        pass

    def _resolve_url(self) -> str:
        """Resolve which of the candidate base urls to use."""
        for url in self._connection_info["urls"]:
            try:
                print(f"Testing {url}")
                # Original code has auth token passed but is it actually used?
                resp = requests.get(f"{url}/healthcheck")
                print(f"Response code  {resp.status_code}")
                if resp.status_code == 200:
                    print(f"200 status code for {url}")
                    print(f"Response {resp.text}")
                    return url
            except requests.ConnectionError:
                pass
        # Needs better exception message
        raise RuntimeError("None of the Storage URLs provided worked")

    def request(  # type: ignore[override]
        self, method: str, url: str, *args: Any, **kwargs: Any
    ) -> requests.Response:
        """Perform HTTP request with preconfigured base url and auth token."""
        kwargs.setdefault("headers", {})
        kwargs["headers"].update(self._headers)
        return super().request(method, urljoin(self._base_url, url), *args, **kwargs)
