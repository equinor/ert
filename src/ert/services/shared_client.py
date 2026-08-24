from __future__ import annotations

import json
import logging
import os
import ssl
import threading
from os import PathLike
from pathlib import Path
from textwrap import dedent
from typing import Any, ClassVar, Literal

import httpx
from httpx_retries import Retry, RetryTransport
from pydantic import BaseModel, ValidationError

from ert.services.ert_server import create_ert_server_controller

logger = logging.getLogger(__name__)


class ErtClientConnectionInfo(BaseModel, extra="forbid"):
    base_url: str
    auth_token: str | None = None
    cert: str | bool = False


type Methods = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


ENV_VAR = "ERT_STORAGE_CONNECTION_STRING"

# Avoid searching for the connection information on every request. We assume
# that a single client process will only ever want to connect to a single ERT
# Storage server during its lifetime, so we don't provide an API for managing
# this cache.
_CACHED_CONN_INFO: ErtClientConnectionInfo | None = None


class SharedClient:
    """A long-lived thread-safe client for the ERT server."""

    _instance: ClassVar[SharedClient | None] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, project: Path, client: Client) -> None:
        self._project = project
        self._client: Client = client

    @classmethod
    def get_client(
        cls, project: PathLike[str], timeout: int | None = None
    ) -> SharedClient:
        key = Path(project).resolve()
        with cls._instance_lock:
            if cls._instance and not cls._instance._client.is_closed:
                return cls._instance
            client = create_ertserver_client(key, timeout=timeout)
            cls._instance = cls(key, client)
        return cls._instance

    @property
    def project(self) -> Path:
        return self._project

    @property
    def conn_info(self) -> ErtClientConnectionInfo:
        return self._client.conn_info

    def request(
        self,
        method: Methods,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        try:
            return self._client.request(method, url, **kwargs)
        except httpx.HTTPError as e:
            logger.warning(
                dedent(
                    f"""
                    Error occurred for request: [{method.upper()}]<{url}>
                    For project: {self._project}
                    Received {type(e).__name__} with message: {e}
                    """
                )
            )
            raise

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)


class Client(httpx.Client):
    def __init__(self, conn_info: ErtClientConnectionInfo | None = None) -> None:
        if conn_info is None:
            conn_info = find_conn_info()

        self.conn_info = conn_info

        headers = {}
        if conn_info.auth_token is not None:
            headers = {"Token": conn_info.auth_token}
        super().__init__(
            base_url=conn_info.base_url,
            headers=headers,
            transport=RetryTransport(
                httpx.HTTPTransport(
                    verify=ssl.create_default_context(cafile=conn_info.cert)
                    if isinstance(conn_info.cert, str)
                    else conn_info.cert,
                ),
                retry=Retry(total=5, backoff_factor=0.5),
            ),
            timeout=3,
        )


def create_ertserver_client(project: Path, timeout: int | None = None) -> Client:
    """Read connection info from file in path and create HTTP client."""
    controller = create_ert_server_controller(timeout=timeout, project=project)
    info = controller.fetch_connection_info()
    return Client(
        conn_info=ErtClientConnectionInfo(
            base_url=controller.fetch_url(),
            auth_token=controller.fetch_auth()[1],
            cert=info["cert"],
        )
    )


def find_conn_info() -> ErtClientConnectionInfo:
    """
    The base url and auth token are read from either:
    The file `storage_server.json`, starting from the current working directory
    or the environment variable `ERT_STORAGE_CONNECTION_STRING`

    In both cases the configuration is represented by JSON representation of the
    `ConnInfo` pydantic model.

    In the event that nothing is found, a RuntimeError is raised.
    """
    global _CACHED_CONN_INFO  # ruff: ignore[global-statement]
    if _CACHED_CONN_INFO is not None:
        return _CACHED_CONN_INFO

    conn_str = os.environ.get(ENV_VAR)

    # This could be an empty string rather than None, as by the shell
    # invocation: env ERT_STORAGE_CONNECTION_STRING= python
    if not conn_str:
        # Look for `storage_server.json` from cwd up to root.
        root = Path("/")
        path = Path.cwd()
        while path != root:
            try:
                conn_str = (path / "storage_server.json").read_text()
                break
            except FileNotFoundError:
                path = path.parent

    if not conn_str:
        raise RuntimeError("No Storage connection configuration found")

    try:
        conn_info = ErtClientConnectionInfo.model_validate_json(conn_str)
    except (json.JSONDecodeError, ValidationError) as e:
        raise RuntimeError("Invalid storage connection configuration") from e
    else:
        _CACHED_CONN_INFO = conn_info
        return conn_info
