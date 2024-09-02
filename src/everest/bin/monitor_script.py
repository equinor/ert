#!/usr/bin/env python

import argparse
import logging
import signal
import threading
from functools import partial

from everest.config import EverestConfig
from everest.detached import ServerStatus, everserver_status, server_is_running

from .utils import (
    handle_keyboard_interrupt,
    report_on_previous_run,
    run_detached_monitor,
)


def monitor_entry(args=None):
    """Entry point for monitoring an optimization."""
    parser = _build_args_parser()
    options = parser.parse_args(args)

    if options.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Remove the null handler if set:
        logging.getLogger().removeHandler(logging.NullHandler())

    if threading.current_thread() is threading.main_thread():
        signal.signal(
            signal.SIGINT,
            partial(handle_keyboard_interrupt, options=options),
        )

    monitor_everest(options)


def _build_args_parser():
    """Build arg parser"""

    arg_parser = argparse.ArgumentParser(
        description="Everest console monitor a running optimization case based on a config file",
        usage="everest monitor <config_file>",
    )
    arg_parser.add_argument(
        "config",
        type=partial(EverestConfig.load_file_with_argparser, parser=arg_parser),
        help="The path to the everest configuration file",
    )
    arg_parser.add_argument(
        "--debug", action="store_true", help="Display debug information in the terminal"
    )
    arg_parser.add_argument(
        "--show-all-jobs",
        action="store_true",
        help="Display all jobs executed from the forward model",
    )

    return arg_parser


def monitor_everest(options):
    config: EverestConfig = options.config
    server_state = everserver_status(options.config)

    if server_is_running(config):
        run_detached_monitor(config, show_all_jobs=options.show_all_jobs)
        server_state = everserver_status(config)
        if server_state["status"] == ServerStatus.failed:
            raise SystemExit(server_state["message"])
        if server_state["message"] is not None:
            print(server_state["message"])
    elif server_state["status"] == ServerStatus.never_run:
        config_file = config.config_file
        print(
            "The optimization has not run yet.\n"
            "To run the optimization use command:\n"
            f"  `everest run {config_file}`"
        )
    else:
        report_on_previous_run(config)


if __name__ == "__main__":
    monitor_entry()
