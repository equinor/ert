import os
import sys
import uvicorn
import socket
import json
import argparse
from pathlib import Path
from ert_shared import port_handler
from ert_shared import __file__ as ert_shared_path
from ert_shared.plugins import ErtPluginContext
from ert_shared.storage.command import add_parser_options
from uvicorn.supervisors import ChangeReload
from typing import List


class Server(uvicorn.Server):
    def __init__(self, config, connection_info, info_file):
        super().__init__(config)
        self.connection_info = connection_info
        self.info_file = info_file

    async def startup(self, sockets=None):
        """Overridden startup that also sends connection information"""
        await super().startup(sockets)
        if not self.started:
            return
        write_to_pipe(self.connection_info)


def generate_authtoken():
    import random
    import string

    chars = string.ascii_letters + string.digits
    return "".join([random.choice(chars) for _ in range(16)])


def write_to_pipe(connection_info: dict):
    """Write connection information directly to the calling program (ERT) via a
    communication pipe."""
    fd = os.environ.get("ERT_COMM_FD")
    if fd is None:
        return
    with os.fdopen(int(fd), "w") as f:
        f.write(str(connection_info))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    add_parser_options(ap)
    return ap.parse_args()


def find_ert_config() -> str:
    """
    Try to find an ERT config file. Quit if either no files look like ERT
    configs, or if it's ambiguous.
    """
    ert_configs: List[str] = []
    for file_ in os.listdir():
        if not os.path.isfile(file_):
            continue
        if file_.endswith(".ert"):
            ert_configs.append(file_)
    if len(ert_configs) == 1:
        return ert_configs[0]
    if len(ert_configs) == 0:
        sys.exit(f"No ERT configs found in {os.getcwd()}")
    sys.exit(
        f"It is ambigious which ERT config to use in {os.getcwd()}\n"
        "Usage: ert api [CONFIG]\n"
    )


def _create_connection_info(sock, authtoken):
    connection_info = {
        "urls": [
            f"http://{host}:{sock.getsockname()[1]}"
            for host in (
                sock.getsockname()[0],
                socket.gethostname(),
                socket.getfqdn(),
            )
        ],
        "authtoken": authtoken,
    }

    os.environ["ERT_STORAGE_CONNECTION_STRING"] = json.dumps(
        connection_info, separators=(",", ":")
    )

    return connection_info


def run_server(args=None, debug=False):
    if args is None:
        args = parse_args()

    if "ERT_STORAGE_TOKEN" in os.environ:
        authtoken = os.environ["ERT_STORAGE_TOKEN"]
    else:
        authtoken = generate_authtoken()
        os.environ["ERT_STORAGE_TOKEN"] = authtoken

    lockfile = Path.cwd() / "storage_server.json"
    if lockfile.exists():
        sys.exit("'storage_server.json' already exists")

    config_args = {}
    if args.debug or debug:
        config_args.update(reload=True, reload_dirs=[os.path.dirname(ert_shared_path)])
        os.environ["ERT_STORAGE_DEBUG"] = "1"

    _, _, sock = port_handler.find_available_port(custom_host=args.host)

    connection_info = _create_connection_info(sock, authtoken)

    # Appropriated from uvicorn.main:run
    os.environ["ERT_STORAGE_NO_TOKEN"] = "1"
    if args.enable_new_storage:
        args.database_url = "sqlite:///ert.db"
    if args.database_url:
        os.environ["ERT_STORAGE_DATABASE_URL"] = args.database_url
        config = uvicorn.Config("ert_storage.app:app", **config_args)
    else:
        # Dark Storage imports from ERT Storage, which connects to the database
        # at startup. We set the database URL to an SQLite in-memory database so
        # that the import succeeds.
        os.environ["ERT_STORAGE_DATABASE_URL"] = "sqlite://"
        os.environ["ERT_STORAGE_RES_CONFIG"] = args.config or find_ert_config()
        config = uvicorn.Config("ert_shared.dark_storage.app:app", **config_args)
    server = Server(config, json.dumps(connection_info), lockfile)

    print("Storage server is ready to accept requests. Listening on:")
    for url in connection_info["urls"]:
        print(f"  {url}")

    print(f"\nOpenAPI Docs: {url}/docs", file=sys.stderr)
    if args.debug or debug:
        print("\tRunning in NON-SECURE debug mode.\n")
        os.environ["ERT_STORAGE_NO_TOKEN"] = "1"
    else:
        print("\tUsername: __token__")
        print(f"\tPassword: {connection_info['authtoken']}\n")

    if config.should_reload:
        supervisor = ChangeReload(config, target=server.run, sockets=[sock])
        supervisor.run()
    else:
        server.run(sockets=[sock])


def terminate_on_parent_death():
    """Quit the server when the parent does a SIGABRT or is otherwise destroyed.
    This functionality has existed on Linux for a good while, but it isn't
    exposed in the Python standard library. Use ctypes to hook into the
    functionality.
    """
    if sys.platform != "linux" or "ERT_COMM_FD" not in os.environ:
        return

    from ctypes import CDLL, c_int, c_ulong
    import signal

    lib = CDLL(None)

    # from <sys/prctl.h>
    # int prctl(int option, ...)
    prctl = lib.prctl
    prctl.restype = c_int
    prctl.argtypes = (c_int, c_ulong)

    # from <linux/prctl.h>
    PR_SET_PDEATHSIG = 1

    # connect parent death signal to our SIGTERM
    prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


if __name__ == "__main__":
    # pylint: disable=W0611
    import ert_logging

    terminate_on_parent_death()
    with ErtPluginContext() as context:
        run_server(debug=False)
