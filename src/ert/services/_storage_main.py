import argparse
import json
import logging
import logging.config
import os
import random
import signal
import socket
import string
import sys
import warnings
from typing import Any, Dict, List, Optional, Union

import uvicorn
import yaml
from uvicorn.supervisors import ChangeReload

from ert.logging import STORAGE_LOG_CONFIG
from ert.plugins import ErtPluginContext
from ert.shared import __file__ as ert_shared_path
from ert.shared import find_available_socket
from ert.shared.storage.command import add_parser_options


class Server(uvicorn.Server):
    def __init__(
        self,
        config: uvicorn.Config,
        connection_info: Union[str, Dict[str, Any]],
    ):
        super().__init__(config)
        self.connection_info = connection_info

    async def startup(self, sockets: Optional[List[socket.socket]] = None) -> None:
        """Overridden startup that also sends connection information"""
        await super().startup(sockets)
        if not self.started:
            return
        write_to_pipe(self.connection_info)


def generate_authtoken() -> str:
    chars = string.ascii_letters + string.digits
    return "".join([random.choice(chars) for _ in range(16)])


def write_to_pipe(connection_info: Union[str, Dict[str, Any]]) -> None:
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


def _create_connection_info(sock: socket.socket, authtoken: str) -> Dict[str, Any]:
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


def run_server(args: Optional[argparse.Namespace] = None, debug: bool = False) -> None:
    if args is None:
        args = parse_args()

    if "ERT_STORAGE_TOKEN" in os.environ:
        authtoken = os.environ["ERT_STORAGE_TOKEN"]
    else:
        authtoken = generate_authtoken()
        os.environ["ERT_STORAGE_TOKEN"] = authtoken

    config_args: Dict[str, Any] = {}
    if args.debug or debug:
        config_args.update(reload=True, reload_dirs=[os.path.dirname(ert_shared_path)])
        os.environ["ERT_STORAGE_DEBUG"] = "1"

    sock = find_available_socket(custom_host=args.host)
    connection_info = _create_connection_info(sock, authtoken)

    # Appropriated from uvicorn.main:run
    os.environ["ERT_STORAGE_NO_TOKEN"] = "1"
    os.environ["ERT_STORAGE_ENS_PATH"] = os.path.abspath(args.project)
    config = uvicorn.Config("ert.dark_storage.app:app", **config_args)
    server = Server(config, json.dumps(connection_info))

    logger = logging.getLogger("ert.shared.storage.info")
    log_level = logging.INFO if args.verbose else logging.WARNING
    logger.setLevel(log_level)
    logger.info("Storage server is ready to accept requests. Listening on:")
    for url in connection_info["urls"]:
        logger.info(f"  {url}")
        logger.info(f"\nOpenAPI Docs: {url}/docs")

    if args.debug or debug:
        logger.info("\tRunning in NON-SECURE debug mode.\n")
        os.environ["ERT_STORAGE_NO_TOKEN"] = "1"

    if config.should_reload:
        supervisor = ChangeReload(config, target=server.run, sockets=[sock])
        supervisor.run()
    else:
        server.run(sockets=[sock])


def terminate_on_parent_death() -> None:
    """Quit the server when the parent does a SIGABRT or is otherwise destroyed.
    This functionality has existed on Linux for a good while, but it isn't
    exposed in the Python standard library. Use ctypes to hook into the
    functionality.
    """
    if sys.platform != "linux" or "ERT_COMM_FD" not in os.environ:
        return

    from ctypes import CDLL, c_int, c_ulong  # noqa: PLC0415

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
    with open(STORAGE_LOG_CONFIG, encoding="utf-8") as conf_file:
        logging_conf = yaml.safe_load(conf_file)
        logging.config.dictConfig(logging_conf)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    uvicorn.config.LOGGING_CONFIG.clear()
    uvicorn.config.LOGGING_CONFIG.update(logging_conf)
    terminate_on_parent_death()
    with ErtPluginContext(logger=logging.getLogger()) as context:
        run_server(debug=False)
