#!/usr/bin/env python

import argparse
import signal
import threading
from functools import partial

from everest.config import EverestConfig, ServerConfig
from everest.detached import ExperimentState, everserver_status, server_is_running
from everest.everest_storage import EverestStorage

from .utils import (
    handle_keyboard_interrupt,
    report_on_previous_run,
    run_detached_monitor,
    setup_logging,
)


def monitor_entry(args: list[str] | None = None) -> None:
    """Entry point for monitoring an optimization."""
    parser = _build_args_parser()
    options = parser.parse_args(args)

    with setup_logging(options):
        if threading.current_thread() is threading.main_thread():
            signal.signal(
                signal.SIGINT,
                partial(handle_keyboard_interrupt, options=options),
            )

        EverestStorage.check_for_deprecated_seba_storage(
            options.config.optimization_output_dir
        )

        monitor_everest(options)


def _build_args_parser() -> argparse.ArgumentParser:
    """Build arg parser"""

    arg_parser = argparse.ArgumentParser(
        description=(
            "Everest console monitor a running optimization case based on a config file"
        ),
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
        help=(
            "This option no longer has an effect, "
            "and will be removed in a future version"
        ),
    )

    return arg_parser


def monitor_everest(options: argparse.Namespace) -> None:
    config: EverestConfig = options.config
    status_path = ServerConfig.get_everserver_status_path(config.output_dir)
    server_state = everserver_status(status_path)
    server_context = ServerConfig.get_server_context(config.output_dir)
    if server_is_running(*server_context):
        run_detached_monitor(server_context=server_context)
        server_state = everserver_status(status_path)
        if server_state["status"] == ExperimentState.failed:
            raise SystemExit(server_state["message"])
        if server_state["message"] is not None:
            print(server_state["message"])
    elif server_state["status"] == ExperimentState.never_run:
        config_file = config.config_file
        print(
            "The optimization has not run yet.\n"
            "To run the optimization use command:\n"
            f"  `everest run {config_file}`"
        )
    else:
        report_on_previous_run(
            config_file=config.config_file,
            everserver_status_path=status_path,
            optimization_output_dir=config.optimization_output_dir,
        )


if __name__ == "__main__":
    monitor_entry()
