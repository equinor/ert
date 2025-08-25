#!/usr/bin/env python
import argparse
import asyncio
import json
import logging
import logging.config
import os
import signal
import socket
import threading
from functools import partial

from _ert.threading import ErtThread
from ert.config import QueueSystem
from everest.config import EverestConfig, ServerConfig
from everest.detached import (
    ExperimentState,
    everserver_status,
    server_is_running,
    start_experiment,
    start_server,
    wait_for_server,
)
from everest.everest_storage import EverestStorage
from everest.util import (
    makedirs_if_needed,
    version_info,
    warn_user_that_runpath_is_nonempty,
)

from .utils import (
    handle_keyboard_interrupt,
    report_on_previous_run,
    run_detached_monitor,
    run_empty_detached_monitor,
    setup_logging,
    show_scaled_controls_warning,
)

logger = logging.getLogger(__name__)


def everest_entry(args: list[str] | None = None) -> None:
    """Entry point for running an optimization."""
    parser = _build_args_parser()

    options = parser.parse_args(args)

    makedirs_if_needed(options.config.output_dir, roll_if_exists=True)
    with setup_logging(options):
        logger.info(version_info())

        client_machine_hostname = socket.gethostname()
        server_queue_system = options.config.server.queue_system.name
        simulator_queue_system = options.config.simulator.queue_system.name

        server_info_str = "The optimization will be run by an experiment server on " + (
            f"this machine ({client_machine_hostname}). "
            f"Pressing Ctrl+C will stop the optimization and exit."
            if server_queue_system == QueueSystem.LOCAL
            else f"the {server_queue_system} queue."
        )

        simulator_info_str = (
            "The experiment server will submit the ERT forward model to run on "
        ) + (
            f"this machine ({client_machine_hostname})"
            if simulator_queue_system == QueueSystem.LOCAL
            else f"the {simulator_queue_system} queue."
        )

        print(
            "=======You are now running everest=======\n"
            f"* Monitoring from this machine: {client_machine_hostname}.\n"
            f"* {server_info_str}\n"
            f"* {simulator_info_str}\n"
            "=========================================\n"
            + (
                ""
                if server_queue_system == QueueSystem.LOCAL
                else "*Since the server is running on the queue, "
                "pressing Ctrl+C will NOT stop the optimization, it will "
                f"only shut down the monitoring on this "
                f"machine ({client_machine_hostname}).\n"
            ),
        )

        logger.info(
            f"server runs on {server_queue_system}, "
            f"simulator runs on {simulator_queue_system}"
        )

        if threading.current_thread() is threading.main_thread():
            signal.signal(
                signal.SIGINT,
                partial(handle_keyboard_interrupt, options=options),
            )

        asyncio.run(run_everest(options))


def _build_args_parser() -> argparse.ArgumentParser:
    """Build arg parser"""

    arg_parser = argparse.ArgumentParser(
        description=(
            "Everest console runner, start an "
            "optimization case based on a given config file"
        ),
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
        "--gui",
        action="store_true",
        help="Spawn a GUI monitoring simulation statuses.",
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
    arg_parser.add_argument(
        "--skip-prompt",
        action="store_true",
        help="Flag used to disable user prompts that will stop execution.",
    )
    arg_parser.add_argument(
        "--disable-monitoring",
        action="store_true",
        help=(
            "Disable monitoring of the optimization run. "
            "This will reduce the output to the terminal."
        ),
    )
    return arg_parser


async def run_everest(options: argparse.Namespace) -> None:
    everserver_status_path = ServerConfig.get_everserver_status_path(
        options.config.output_dir
    )

    if not options.new_run:
        EverestStorage.check_for_deprecated_seba_storage(
            options.config.optimization_output_dir
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
    elif server_state["status"] == ExperimentState.never_run or options.new_run:
        config_dict = options.config.to_dict()
        logger.info("Running everest with the following config:")
        logger.info(json.dumps(config_dict, sort_keys=True, indent=2))
        for fm_job in options.config.forward_model_step_commands:
            job_name = fm_job.split()[0]
            logger.info(f"Everest forward model contains job {job_name}")

        if (
            options.config.simulation_dir is not None
            and os.path.exists(options.config.simulation_dir)
            and any(os.listdir(options.config.simulation_dir))
        ):
            warn_user_that_runpath_is_nonempty()
        if not options.skip_prompt:
            show_scaled_controls_warning()

        try:
            output_dir = options.config.output_dir
            config_file = options.config.config_file
            save_config_path = os.path.join(output_dir, config_file)
            options.config.dump(save_config_path)
        except (OSError, LookupError) as e:
            logger.error(f"Failed to save optimization config: {e}")

        logging_level = logging.DEBUG if options.debug else options.config.logging_level

        print("Adding everest server to queue ...")
        logger.debug("Submitting everserver")
        try:
            await asyncio.wait_for(
                start_server(options.config, logging_level), timeout=1800
            )  # 30 minutes
            logger.debug("Everserver submitted and started")
        except TimeoutError as e:
            logger.error("Everserver failed to start within timeout")
            raise SystemExit("Failed to start the server") from e

        print("Waiting for server ...")
        logger.debug("Waiting for response from everserver")
        wait_for_server(options.config.output_dir, timeout=600)
        print("Everest server found!")
        logger.info("Got response from everserver. Starting experiment")

        start_experiment(
            server_context=ServerConfig.get_server_context(options.config.output_dir),
            config=options.config,
        )

        # blocks until the run is finished
        if options.gui:
            from everest.gui.main import run_gui  # noqa

            monitor_thread = ErtThread(
                target=run_empty_detached_monitor
                if options.disable_monitoring
                else run_detached_monitor,
                name="Everest CLI monitor thread",
                args=[ServerConfig.get_server_context(options.config.output_dir)],
                daemon=True,
            )
            monitor_thread.start()
            run_gui(options.config.output_dir)
            monitor_thread.join()
        elif options.disable_monitoring:
            run_empty_detached_monitor(
                server_context=ServerConfig.get_server_context(
                    options.config.output_dir
                )
            )
        else:
            run_detached_monitor(
                server_context=ServerConfig.get_server_context(
                    options.config.output_dir
                )
            )

        logger.info("Everest experiment finished")

        server_state = everserver_status(everserver_status_path)
        server_state_info = server_state["message"]
        if server_state["status"] == ExperimentState.failed:
            raise SystemExit(f"Everest run failed with: {server_state_info}")
        if server_state_info is not None:
            logger.info(f"Everest run finished with: {server_state_info}")
            print(server_state_info)
    else:
        report_on_previous_run(
            config_file=options.config.config_file,
            everserver_status_path=everserver_status_path,
            optimization_output_dir=options.config.optimization_output_dir,
        )


if __name__ == "__main__":
    everest_entry()
