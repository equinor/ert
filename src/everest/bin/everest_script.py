#!/usr/bin/env python

import argparse
import json
import logging
import signal
import threading
from functools import partial

from ert.config import ErtConfig
from ert.storage import open_storage
from everest.config import EverestConfig
from everest.detached import (
    ServerStatus,
    everserver_status,
    generate_everserver_ert_config,
    server_is_running,
    start_server,
    wait_for_context,
    wait_for_server,
)
from everest.plugins.site_config_env import PluginSiteConfigEnv
from everest.util import makedirs_if_needed, version_info

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

    run_everest(options)


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


def _run_everest(options, ert_config, storage):
    with PluginSiteConfigEnv():
        context = start_server(options.config, ert_config, storage)
        print("Waiting for server ...")
        wait_for_server(options.config, timeout=600, context=context)
        print("Everest server found!")
        run_detached_monitor(options.config, show_all_jobs=options.show_all_jobs)
        wait_for_context()


def run_everest(options):
    logger = logging.getLogger("everest_main")
    server_state = everserver_status(options.config)

    if server_is_running(options.config):
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

        with PluginSiteConfigEnv():
            ert_config = ErtConfig.with_plugins().from_dict(
                config_dict=generate_everserver_ert_config(
                    options.config, options.debug
                )
            )

        makedirs_if_needed(options.config.output_dir, roll_if_exists=True)

        with open_storage(ert_config.ens_path, "w") as storage:
            _run_everest(options, ert_config, storage)

        server_state = everserver_status(options.config)
        server_state_info = server_state["message"]
        if server_state["status"] == ServerStatus.failed:
            logger.error("Everest run failed with: {}".format(server_state_info))
            raise SystemExit(server_state_info)
        if server_state_info is not None:
            logger.info("Everest run finished with: {}".format(server_state_info))
            print(server_state_info)
    else:
        report_on_previous_run(options.config)


if __name__ == "__main__":
    everest_entry()
