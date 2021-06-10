import os
import sys
import uvicorn
import socket
import json
import argparse
from pathlib import Path
from ert_shared import port_handler
from ert_shared import __file__ as ert_shared_path
from ert_shared.storage import connection
from ert_shared.storage.command import add_parser_options
from uvicorn.supervisors import ChangeReload


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
        write_to_file(self.connection_info, self.info_file)

    async def shutdown(self, sockets=None):
        """Overridden shutdown that deletes the lockfile"""
        await super().shutdown(sockets)
        self.info_file.unlink()


def generate_authtoken():
    import random
    import string

    chars = string.ascii_letters + string.digits
    return "".join([random.choice(chars) for _ in range(16)])


def write_to_file(connection_info: dict, lockfile):
    """Write connection information to 'storage_server.json'"""
    lockfile.write_text(connection_info)
    connection.set_global_info(os.getcwd())


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


def run_server(args=None, debug=False):
    if args is None:
        args = parse_args()

    if "ERT_STORAGE_TOKEN" in os.environ:
        authtoken = os.environ["ERT_STORAGE_TOKEN"]
    else:
        authtoken = generate_authtoken()
        os.environ["ERT_STORAGE_TOKEN"] = authtoken

    # Use sqlite in cwd if nothing else is specified
    if "ERT_STORAGE_DATABASE_URL" not in os.environ:
        os.environ["ERT_STORAGE_DATABASE_URL"] = "sqlite:///ert.db"

    lockfile = Path.cwd() / "storage_server.json"
    if lockfile.exists():
        sys.exit("'storage_server.json' already exists")

    config_args = {}
    if args.debug or debug:
        config_args.update(reload=True, reload_dirs=[os.path.dirname(ert_shared_path)])
        os.environ["ERT_STORAGE_DEBUG"] = "1"

    host, port = port_handler.find_available_port(custom_host=args.host)
    sock = port_handler.get_socket(host=host, port=port)
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

    # Appropriated from uvicorn.main:run
    config = uvicorn.Config("ert_storage.app:app", **config_args)
    server = Server(config, json.dumps(connection_info), lockfile)

    print("Storage server is ready to accept requests. Listening on:")
    for url in connection_info["urls"]:
        print(f"  {url}")

    print(f"\nOpenAPI Docs: {url}/docs", file=sys.stderr)
    if args.debug or debug:
        print("\tRunning in NON-SECURE debug mode.\n")
    else:
        print(f"\tUsername: __token__")
        print(f"\tPassword: {connection_info['authtoken']}\n")

    if config.should_reload:
        supervisor = ChangeReload(config, target=server.run, sockets=[sock])
        supervisor.run()
    else:
        server.run(sockets=[sock])
