import json
import os
from pathlib import Path

from pydantic import BaseModel, ValidationError


class ConnInfo(BaseModel):
    base_url: str
    auth_token: str | None = None
    cert: str | bool = False


ENV_VAR = "ERT_STORAGE_CONNECTION_STRING"

# Avoid searching for the connection information on every request. We assume
# that a single client process will only ever want to connect to a single ERT
# Storage server during its lifetime, so we don't provide an API for managing
# this cache.
_CACHED_CONN_INFO: ConnInfo | None = None


def find_conn_info() -> ConnInfo:
    """
    The base url and auth token are read from either:
    The file `storage_server.json`, starting from the current working directory
    or the environment variable `ERT_STORAGE_CONNECTION_STRING`

    In both cases the configuration is represented by JSON representation of the
    `ConnInfo` pydantic model.

    In the event that nothing is found, a RuntimeError is raised.
    """
    global _CACHED_CONN_INFO  # noqa: PLW0603
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
        conn_info = ConnInfo.model_validate_json(conn_str)
    except (json.JSONDecodeError, ValidationError) as e:
        raise RuntimeError("Invalid storage connection configuration") from e
    else:
        _CACHED_CONN_INFO = conn_info
        return conn_info
