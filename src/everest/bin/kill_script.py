#!/usr/bin/env python

import argparse
import json
import logging
import os
import signal
import sys
import threading
import traceback
from functools import partial
from typing import Any

from everest.bin.utils import setup_logging
from everest.config import EverestConfig, ServerConfig
from everest.detached import server_is_running, stop_server, wait_for_server_to_stop
from everest.util import version_info

logger = logging.getLogger(__name__)


def kill_entry(args: list[str] | None = None) -> None:
    """Entry point for running an optimization."""
    parser = _build_args_parser()
    options = parser.parse_args(args)
    with setup_logging(options):
        logger.info(version_info())
        logger.debug(json.dumps(options.config.to_dict(), sort_keys=True, indent=2))

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, partial(_handle_keyboard_interrupt))

        kill_everest(options)


def _build_args_parser() -> argparse.ArgumentParser:
    """Build arg parser"""

    arg_parser = argparse.ArgumentParser(
        description="Kill a running optimization case based on a given config file",
        usage="everest kill <config_file>",
    )
    arg_parser.add_argument(
        "config",
        type=partial(EverestConfig.load_file_with_argparser, parser=arg_parser),
        help="The path to the everest configuration file",
    )
    arg_parser.add_argument(
        "--debug", action="store_true", help="Display debug information in the terminal"
    )

    return arg_parser


def _handle_keyboard_interrupt(signal: int, _: Any, after: bool = False) -> None:
    if after:
        print(
            f"KeyboardInterrupt (ID: {signal}) has been caught, "
            "but kill request will proceed..."
        )
    else:
        print(
            f"KeyboardInterrupt (ID: {signal}) has been caught, "
            "kill request will be cancelled..."
        )
    sys.tracebacklimit = 0
    sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa SIM115
    sys.exit()


def kill_everest(options: argparse.Namespace) -> None:
    server_context = ServerConfig.get_server_context(options.config.output_dir)
    if not server_is_running(*server_context):
        print("Server is not running.")
        return

    stopping = stop_server(server_context)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, partial(_handle_keyboard_interrupt, after=True))

    if not stopping:
        print("Stop request failed, you may have to kill the server manually")
        return
    try:
        print("Waiting for server to stop ...")
        wait_for_server_to_stop(server_context, timeout=60)
        print("Server stopped.")
    except Exception:
        logger.debug(traceback.format_exc())
        print(
            "Server is still running after 60 seconds, "
            "you may have to kill the server manually"
        )


if __name__ == "__main__":
    kill_entry()
