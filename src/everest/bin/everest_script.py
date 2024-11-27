#!/usr/bin/env python

import argparse
import asyncio
import json
import logging
import os
import signal
import threading
from functools import partial

from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig, ServerConfig
from everest.detached import (
    ServerStatus,
    everserver_status,
    server_is_running,
    start_server,
    wait_for_server,
)
from everest.simulator.everest_to_ert import everest_to_ert_config
from everest.strings import EVEREST
from everest.util import (
    makedirs_if_needed,
    version_info,
    warn_user_that_runpath_is_nonempty,
)

from .utils import (
    handle_keyboard_interrupt,
    report_on_previous_run,
    run_detached_monitor,
)


def everest_entry(args=None):
    """Entry point for running an optimization."""
    parser = _build_args_parser()
    options = parser.parse_args(args)

    if options.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Remove the null handler if set:
        logging.getLogger().removeHandler(logging.NullHandler())

    logging.info(version_info())
    logging.debug(json.dumps(options.config.to_dict(), sort_keys=True, indent=2))

    if threading.current_thread() is threading.main_thread():
        signal.signal(
            signal.SIGINT,
            partial(handle_keyboard_interrupt, options=options),
        )

    asyncio.run(run_everest(options))


def _build_args_parser():
    """Build arg parser"""

    arg_parser = argparse.ArgumentParser(
        description="Everest console runner, start an optimization case based on a given config file",
        usage="everest run <config_file> [arguments]",
    )
    arg_parser.add_argument(
        "config",
        type=partial(EverestConfig.load_file_with_argparser, parser=arg_parser),
        help="The path to the everest configuration file",
    )
    arg_parser.add_argument(
        "--new-run",
        action="store_true",
        help="Run the optimization even though results are already available",
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


async def run_everest(options):
    logger = logging.getLogger("everest_main")
    everserver_status_path = ServerConfig.get_everserver_status_path(
        options.config.output_dir
    )
    server_state = everserver_status(everserver_status_path)
    if server_is_running(*ServerConfig.get_server_context(options.config.output_dir)):
        config_file = options.config.config_file
        print(
            "An optimization is currently running.\n"
            "To monitor the running optimization use command:\n"
            f"  `everest monitor {config_file}`\n"
            "To kill the running optimization use command:\n"
            f"  `everest kill {config_file}`"
        )
    elif server_state["status"] == ServerStatus.never_run or options.new_run:
        config_dict = options.config.to_dict()
        logger.info("Running everest with config info\n {}".format(config_dict))
        for fm_job in options.config.forward_model or []:
            job_name = fm_job.split()[0]
            logger.info("Everest forward model contains job {}".format(job_name))

        makedirs_if_needed(options.config.output_dir, roll_if_exists=True)

        # Validate ert config
        try:
            _ = everest_to_ert_config(options.config)
        except ValueError as exc:
            raise SystemExit(f"Config validation error: {exc}") from exc

        if EverestRunModel.create(options.config).check_if_runpath_exists():
            warn_user_that_runpath_is_nonempty()

        try:
            output_dir = options.config.output_dir
            config_file = options.config.config_file
            save_config_path = os.path.join(output_dir, config_file)
            options.config.dump(save_config_path)
        except (OSError, LookupError) as e:
            logging.getLogger(EVEREST).error(
                "Failed to save optimization config: {}".format(e)
            )
        await start_server(options.config, options.debug)
        print("Waiting for server ...")
        wait_for_server(options.config.output_dir, timeout=600)
        print("Everest server found!")
        run_detached_monitor(
            server_context=ServerConfig.get_server_context(options.config.output_dir),
            optimization_output_dir=options.config.optimization_output_dir,
            show_all_jobs=options.show_all_jobs,
        )

        server_state = everserver_status(everserver_status_path)
        server_state_info = server_state["message"]
        if server_state["status"] == ServerStatus.failed:
            logger.error("Everest run failed with: {}".format(server_state_info))
            raise SystemExit(server_state_info)
        if server_state_info is not None:
            logger.info("Everest run finished with: {}".format(server_state_info))
            print(server_state_info)
    else:
        report_on_previous_run(
            config_file=options.config.config_file,
            everserver_status_path=everserver_status_path,
            optimization_output_dir=options.config.optimization_output_dir,
        )


if __name__ == "__main__":
    everest_entry()
