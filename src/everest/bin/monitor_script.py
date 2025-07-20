#!/usr/bin/env python

import argparse
import signal
import threading
from functools import partial
from pathlib import Path
from textwrap import dedent

from ert.services import create_ertserver_client
from ert.storage import ErtStorageException, ExperimentState
from everest.config import EverestConfig, ServerConfig
from everest.everest_storage import EverestStorage

from .utils import (
    ArgParseFormatter,
    get_experiment_status,
    handle_keyboard_interrupt,
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
        description=dedent(
            """
            Monitor a running optimization process.

            Closing the console or interrupting the `everest run` process does
            not terminate the optimization process. To continue monitoring the
            running optimization, use `everest monitor config_file.yml`. To stop
            a running optimization, use `everest kill config_file.yml`.
            """
        ),
        formatter_class=ArgParseFormatter,
        usage="everest monitor <config_file>",
    )
    arg_parser.add_argument(
        "config",
        type=partial(EverestConfig.load_file_with_argparser, parser=arg_parser),
        help="The path to the everest configuration file.",
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Display debug information in the terminal.",
    )
    arg_parser.add_argument(
        "--show-all-jobs",
        action="store_true",
        help=(
            "This option no longer has an effect, "
            "and will be removed in a future version."
        ),
    )

    return arg_parser


def monitor_everest(options: argparse.Namespace) -> None:
    config: EverestConfig = options.config
    try:
        with create_ertserver_client(
            Path(ServerConfig.get_session_dir(config.output_dir)), timeout=1
        ) as client:
            server_context = ServerConfig.get_server_context_from_conn_info(
                client.conn_info
            )
            run_detached_monitor(server_context=server_context)

            try:
                experiment_status = get_experiment_status(str(config.storage_dir))
                if (
                    experiment_status
                    and experiment_status.status == ExperimentState.failed
                ):
                    raise SystemExit(experiment_status.message or "Optimization failed")
                if experiment_status:
                    print(
                        experiment_status.message
                        or "Optimization completed successfully"
                    )
            except ErtStorageException as err:
                print(f"Error reading experiment status: {err}")

    except TimeoutError:
        try:
            experiment_status = get_experiment_status(str(config.storage_dir))
            if (
                experiment_status is None
                or experiment_status.status == ExperimentState.never_run
            ):
                print(
                    "The optimization has not run yet.\n"
                    "To run the optimization use command:\n"
                    f"  `everest run {config.config_file}`"
                )
            elif experiment_status.status == ExperimentState.failed:
                print(
                    "Optimization run failed, with error: "
                    f"{experiment_status.message}\n"
                )
            else:
                print(
                    "Optimization already completed.\n"
                    f"Results are stored in {config.optimization_output_dir}"
                )
        except ErtStorageException as err:
            print(f"Error reading experiment status: {err}")


if __name__ == "__main__":
    monitor_entry()
