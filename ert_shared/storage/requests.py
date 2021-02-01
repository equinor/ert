from typing import Any, Optional
from pathlib import Path
import requests

from ert_shared.storage.connection import get_info


_conn_info: Optional[dict] = None


def connect(project_path: Optional[Path] = None) -> None:
    """
    Initialise the Requests instance to connect to the project's storage
    """
    global _conn_info
    _conn_info = get_info(project_path)


def request(method: str, path: str, **kwargs) -> requests.Response:
    if _conn_info is None:
        raise ValueError("A connection to ERT Storage wasn't initialized")

    headers = kwargs.setdefault("headers", {})
    headers["X-Token"] = _conn_info["token"]
    if method not in {"get", "head"}:
        headers.setdefault("Content-Type", "application/json")
    return requests.request(method, f"{_conn_info['baseurl']}/{path}", **kwargs)


def get(path: str, params: Optional[dict] = None, **kwargs) -> requests.Response:
    """Sends a GET request using the requests library"""
    kwargs.setdefault("allow_redirects", True)
    return request("get", path, params=params, **kwargs)


def options(path: str, **kwargs) -> requests.Response:
    """Sends an OPTIONS request using the requests library"""
    kwargs.setdefault("allow_redirects", True)
    return request("options", path, **kwargs)


def head(path: str, **kwargs) -> requests.Response:
    """Sends a HEAD request using the requests library"""
    kwargs.setdefault("allow_redirects", False)
    return request("head", path, **kwargs)


def post(
    path: str, data: Optional[Any] = None, json: Optional[Any] = None, **kwargs
) -> requests.Response:
    """Sends a POST request using the requests library"""
    return request("post", path, data=data, json=json, **kwargs)


def put(path: str, data: Optional[Any] = None, **kwargs) -> requests.Response:
    """Sends a PUT request using the requests library"""
    return request("put", path, data=data, **kwargs)


def patch(path: str, data: Optional[Any] = None, **kwargs) -> requests.Response:
    """Sends a PATCH request using the requests library"""
    return request("patch", path, data=data, **kwargs)


def delete(path: str, **kwargs) -> requests.Response:
    """Sends a DELETE request using the requests library"""
    return request("delete", path, **kwargs)
