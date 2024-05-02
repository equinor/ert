#!/usr/bin/env python

import argparse
import json
import logging
import os
import signal
import sys
import traceback
from functools import partial
import threading

from everest.config import EverestConfig
from everest.detached import server_is_running, stop_server, wait_for_server_to_stop
from everest.util import version_info


def kill_entry(args=None):
    """Entry point for running an optimization."""
    options = setup_args(args)

    if options.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Remove the null handler if set:
        logging.getLogger().removeHandler(logging.NullHandler())

    logging.info(version_info())
    logging.debug(json.dumps(options.config.to_dict(), sort_keys=True, indent=2))

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, partial(_handle_keyboard_interrupt))

    kill_everest(options)


def setup_args(argv):
    """Parse the given argv and return the options object."""

    arg_parser = argparse.ArgumentParser(
        description="Everest console runner",
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

    return arg_parser.parse_args(args=argv)


def _handle_keyboard_interrupt(signal, frame, after=False):
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
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    sys.exit()


def kill_everest(options):
    if not server_is_running(options.config):
        print("Server is not running.")
        return

    stopping = stop_server(options.config)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, partial(_handle_keyboard_interrupt, after=True))

    if not stopping:
        print("Stop request failed, you may have to kill the server manually")
        return
    try:
        print("Waiting for server to stop ...")
        wait_for_server_to_stop(options.config, timeout=60)
        print("Server stopped.")
    except:  # noqa E722
        logging.debug(traceback.format_exc())
        print(
            "Server is still running after 60 seconds, "
            "you may have to kill the server manually"
        )


if __name__ == "__main__":
    kill_entry()
