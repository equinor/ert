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
import threading
import time
import warnings
from types import FrameType
from typing import Any

import uvicorn
import yaml
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from uvicorn.supervisors import ChangeReload

from ert.logging import STORAGE_LOG_CONFIG
from ert.plugins import ErtPluginContext
from ert.shared import __file__ as ert_shared_path
from ert.shared import find_available_socket
from ert.shared.storage.command import add_parser_options
from ert.trace import tracer, tracer_provider

DARK_STORAGE_APP = "ert.dark_storage.app:app"


class Server(uvicorn.Server):
    def __init__(
        self,
        config: uvicorn.Config,
        connection_info: str | dict[str, Any],
    ):
        super().__init__(config)
        self.connection_info = connection_info

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        """Overridden startup that also sends connection information"""
        await super().startup(sockets)
        if not self.started:
            return
        write_to_pipe(self.connection_info)


def generate_authtoken() -> str:
    chars = string.ascii_letters + string.digits
    return "".join([random.choice(chars) for _ in range(16)])


def write_to_pipe(connection_info: str | dict[str, Any]) -> None:
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


def _create_connection_info(sock: socket.socket, authtoken: str) -> dict[str, Any]:
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


def run_server(
    args: argparse.Namespace | None = None,
    debug: bool = False,
    uvicorn_config: uvicorn.Config | None = None,
) -> None:
    if args is None:
        args = parse_args()

    if "ERT_STORAGE_TOKEN" in os.environ:
        authtoken = os.environ["ERT_STORAGE_TOKEN"]
    else:
        authtoken = generate_authtoken()
        os.environ["ERT_STORAGE_TOKEN"] = authtoken

    config_args: dict[str, Any] = {}
    if args.debug or debug:
        config_args.update(reload=True, reload_dirs=[os.path.dirname(ert_shared_path)])
        os.environ["ERT_STORAGE_DEBUG"] = "1"

    sock = find_available_socket(
        custom_host=args.host, custom_range=range(51850, 51870)
    )
    connection_info = _create_connection_info(sock, authtoken)

    # Appropriated from uvicorn.main:run
    os.environ["ERT_STORAGE_NO_TOKEN"] = "1"
    os.environ["ERT_STORAGE_ENS_PATH"] = os.path.abspath(args.project)
    config = (
        uvicorn.Config(DARK_STORAGE_APP, **config_args)
        if uvicorn_config is None
        else uvicorn_config
    )  # uvicorn.Config() resets the logging config (overriding additional handlers added to loggers like e.g. the ert_azurelogger handler added through the pluggin system
    server = Server(config, json.dumps(connection_info))

    logger = logging.getLogger("ert.shared.storage.info")
    if args.verbose and logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
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


def terminate_on_parent_death(
    stopped: threading.Event, parent: int, poll_interval: float = 1.0
) -> None:
    """
    Quit the server when the parent process is no longer running.
    """

    def check_parent_alive() -> bool:
        return os.getppid() == parent

    while check_parent_alive():
        if stopped.is_set():
            return
        time.sleep(poll_interval)

    # Parent is no longer alive, terminate this process.
    os.kill(os.getpid(), signal.SIGTERM)


def main() -> None:
    args = parse_args()
    config_args: dict[str, Any] = {}
    with open(STORAGE_LOG_CONFIG, encoding="utf-8") as conf_file:
        logging_conf = yaml.safe_load(conf_file)
        logging.config.dictConfig(logging_conf)
        config_args.update(log_config=logging_conf)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    if args.debug:
        config_args.update(reload=True, reload_dirs=[os.path.dirname(ert_shared_path)])
    uvicorn_config = uvicorn.Config(
        DARK_STORAGE_APP, **config_args
    )  # Need to run uvicorn.Config before entering the ErtPluginContext because uvicorn.Config overrides the configuration of existing loggers, thus removing log handlers added by ErtPluginContext

    ctx = (
        TraceContextTextMapPropagator().extract(
            carrier={"traceparent": args.traceparent}
        )
        if args.traceparent
        else None
    )

    stopped = threading.Event()
    terminate_on_parent_death_thread = threading.Thread(
        target=terminate_on_parent_death, args=[stopped, args.parent_pid, 1.0]
    )
    with ErtPluginContext(logger=logging.getLogger(), trace_provider=tracer_provider):
        terminate_on_parent_death_thread.start()
        with tracer.start_as_current_span("run_storage_server", ctx):
            logger = logging.getLogger("ert.shared.storage.info")
            try:
                logger.info("Starting dark storage")
                logger.info(f"Started dark storage with parent {args.parent_pid}")
                run_server(args, debug=False, uvicorn_config=uvicorn_config)
            except SystemExit:
                logger.info("Stopping dark storage")
            finally:
                stopped.set()
                terminate_on_parent_death_thread.join()


def sigterm_handler(_signo: int, _stack_frame: FrameType | None) -> None:
    sys.exit(0)


signal.signal(signal.SIGTERM, sigterm_handler)


if __name__ == "__main__":
    main()
