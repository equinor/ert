#!/usr/bin/env python
from __future__ import annotations

import contextlib
import logging
import os
import queue
import sys
from collections import Counter
from typing import TextIO

from _ert.threading import ErtThread
from ert.cli.monitor import Monitor
from ert.cli.workflow import execute_workflow
from ert.config import ErtConfig, QueueSystem
from ert.ensemble_evaluator import EndEvent, EvaluatorServerConfig
from ert.mode_definitions import (
    ENIF_MODE,
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    TEST_RUN_MODE,
    WORKFLOW_MODE,
)
from ert.namespace import Namespace
from ert.plugins import ErtPluginManager
from ert.run_models.event import StatusEvents
from ert.run_models.model_factory import create_model
from ert.storage import open_storage
from ert.storage.local_storage import local_storage_set_ert_config


class ErtCliError(Exception):
    """Used for exception where the Ert command line interface should
    exit with a nonzero return code"""


def run_cli(args: Namespace, plugin_manager: ErtPluginManager | None = None) -> None:
    ert_dir = os.path.abspath(os.path.dirname(args.config))
    os.chdir(ert_dir)
    # Changing current working directory means we need to update
    # the config file to be the base name of the original config
    args.config = os.path.basename(args.config)

    ert_config = ErtConfig.with_plugins().from_file(args.config)

    local_storage_set_ert_config(ert_config)
    counter_fm_steps = Counter(fms.name for fms in ert_config.forward_model_steps)

    # Create logger inside function to make sure all handlers have been added to
    # the root-logger.
    logger = logging.getLogger(__name__)
    for fm_step_name, count in counter_fm_steps.items():
        logger.info(
            f"Config contains forward model step {fm_step_name} {count} time(s)",
        )

    if not ert_config.observations and args.mode not in {
        ENSEMBLE_EXPERIMENT_MODE,
        TEST_RUN_MODE,
        WORKFLOW_MODE,
    }:
        raise ErtCliError(
            f"To run {args.mode}, observations are needed.\n"
            f"Please add an observation file to {args.config}. Example:\n"
            f"'OBS_CONFIG observation_file.txt'."
        )

    if args.mode in {
        ENSEMBLE_SMOOTHER_MODE,
        ENIF_MODE,
        ES_MDA_MODE,
    }:
        if not ert_config.ensemble_config.parameter_configs:
            raise ErtCliError(
                f"To run {args.mode}, GEN_KW, FIELD or SURFACE parameters are "
                f"needed.\nPlease add to file {args.config}"
            )
        if not any(
            p.update for p in ert_config.ensemble_config.parameter_configs.values()
        ):
            raise ErtCliError(
                f"All parameters are set to UPDATE:FALSE in {args.config}"
            )

    if args.mode == WORKFLOW_MODE:
        with open_storage(ert_config.ens_path, "w") as storage:
            execute_workflow(ert_config, storage, args.name)
        return

    status_queue: queue.SimpleQueue[StatusEvents] = queue.SimpleQueue()
    try:
        model = create_model(
            ert_config,
            args,
            status_queue,
        )
    except ValueError as e:
        raise ErtCliError(f"{args.mode} was not valid, failed with: {e}") from e

    if (
        (
            ert_config.analysis_config.design_matrix is not None
            and ert_config.analysis_config.design_matrix.active_realizations is not None
        )
        and hasattr(args, "realizations")
        and args.realizations is not None
    ):
        print(
            "Using realizations intersected between realizations specified "
            f"and DESIGN_MATRIX ({model.active_realizations.count(True)})"
        )

    if args.port_range is None and model.queue_system == QueueSystem.LOCAL:
        # This is within the range for ephemeral ports as defined by
        # most unix flavors https://en.wikipedia.org/wiki/Ephemeral_port
        args.port_range = range(49152, 51819)

    use_ipc_protocol = model.queue_system == QueueSystem.LOCAL
    evaluator_server_config = EvaluatorServerConfig(
        port_range=None
        if args.port_range is None
        else (min(args.port_range), max(args.port_range) + 1),
        use_ipc_protocol=use_ipc_protocol,
    )

    if model.check_if_runpath_exists():
        print(
            "Warning: ERT is running in an existing runpath.\n"
            "Please be aware of the following:\n"
            "- Previously generated results "
            "might be overwritten.\n"
            "- Previously generated files might "
            "be used if not configured correctly.\n"
            f"- {model.get_number_of_existing_runpaths()} out of "
            f"{model.get_number_of_active_realizations()} realizations "
            "are running in existing runpaths.\n"
        )
        logger.warning("ERT is running in an existing runpath")

    thread = ErtThread(
        name="ert_cli_simulation_thread",
        target=model.start_simulations_thread,
        args=(evaluator_server_config,),
    )

    with contextlib.ExitStack() as exit_stack:
        out: TextIO
        if args.disable_monitoring:
            out = exit_stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
        else:
            out = sys.stderr
        monitor = Monitor(out=out, color_always=args.color_always)
        thread.start()
        end_event: EndEvent | None = None
        try:
            end_event = monitor.monitor(
                status_queue, ert_config.analysis_config.log_path
            )
        except (SystemExit, KeyboardInterrupt, OSError):
            print("\nKilling simulations...")
            model.cancel()

    thread.join()

    if end_event is not None and end_event.failed:
        # If monitor has not reported, give some info if the job failed
        msg = end_event.msg if args.disable_monitoring else ""
        raise ErtCliError(msg)
