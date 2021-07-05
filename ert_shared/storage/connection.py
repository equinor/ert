import os
import json
import requests
import getpass
from typing import Dict, Union, Any, Optional
from pathlib import Path


def _json_path(project_path: Union[str, Path] = None) -> Path:
    """Return a path to `storage_server.json`. If `project_path` is None, look for
    any storage server running on the machine.

    """

    if project_path is None:
        if "XDG_RUNTIME_DIR" in os.environ:
            project_path = Path(os.environ["XDG_RUNTIME_DIR"]) / "ert"
            project_path.mkdir(exist_ok=True)
        else:
            project_path = Path(f"/tmp/ert-{getpass.getuser()}")
            project_path.mkdir(mode=0o700, exist_ok=True)

    project_path = Path(project_path)
    return project_path / "storage_server.json"


def get_info(project_path: Union[str, Path] = None) -> Dict[str, Any]:
    """Return a dictionary containing `auth`, a tuple of (username, password) and
    `baseurl`, the URL that the server responds to. If `project_path` is None,
    look for any storage_server running on the machine.

    """

    path = _json_path(project_path)
    if not path.is_file():
        raise RuntimeError(f"{path} is not a file")

    with path.open() as f:
        info = json.load(f)

    auth = ("Token", info["authtoken"])
    for baseurl in info["urls"]:
        try:
            resp = requests.get(f"{baseurl}/healthcheck", auth=auth)
            if resp.status_code == 200:
                break
        except requests.ConnectionError:
            pass
    else:
        raise RuntimeError(f"None of the URLs provided by {path} are valid")

    return {"baseurl": baseurl, "auth": auth}


def set_global_info(project_path: Union[str, Path]):
    """Set `project_path` to be this user's default system-wide ERT project"""
    path = _json_path()
    if path.is_symlink():
        path.unlink()
    path.symlink_to(_json_path(project_path))


def get_project_id() -> Path:
    """Return a path to `storage_server.json`."""

    if "XDG_RUNTIME_DIR" in os.environ:
        project_path = Path(os.environ["XDG_RUNTIME_DIR"]) / "ert"
    else:
        project_path = Path(f"/tmp/ert-{getpass.getuser()}")
    json_path = project_path / "storage_server.json"
    if json_path.exists():
        return json_path.resolve().parent
    raise RuntimeError("No ert storage server found!")


def autostart(project_path: Optional[str] = None) -> str:
    from .server_monitor import ServerMonitor

    # Try to use project specified in args (Defaults to cwd)
    if project_path is not None:
        try:
            get_info(project_path)
            return os.path.realpath(project_path)
        except RuntimeError:
            pass

    # Try to use globally registered ERT Storage
    try:
        path = get_project_id()
        return str(path.absolute())
    except RuntimeError:
        pass

    # Try to start ERT Storage
    monitor = ServerMonitor.get_instance()
    monitor.start()
    monitor.fetch_connection_info()
    return os.path.realpath(os.getcwd())
