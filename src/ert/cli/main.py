#!/usr/bin/env python
from __future__ import annotations

import contextlib
import logging
import os
import queue
import sys
from typing import Any, TextIO

from _ert.threading import ErtThread
from ert.cli import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    TEST_RUN_MODE,
    WORKFLOW_MODE,
)
from ert.cli.model_factory import create_model
from ert.cli.monitor import Monitor
from ert.cli.workflow import execute_workflow
from ert.config import ErtConfig, QueueSystem
from ert.enkf_main import EnKFMain
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.namespace import Namespace
from ert.run_models.base_run_model import StatusEvents
from ert.storage import open_storage
from ert.storage.local_storage import local_storage_set_ert_config


class ErtCliError(Exception):
    pass


class ErtTimeoutError(Exception):
    pass


def run_cli(args: Namespace, _: Any = None) -> None:
    ert_dir = os.path.abspath(os.path.dirname(args.config))
    os.chdir(ert_dir)
    # Changing current working directory means we need to update
    # the config file to be the base name of the original config
    args.config = os.path.basename(args.config)
    ert_config = ErtConfig.from_file(args.config)
    local_storage_set_ert_config(ert_config)

    # Create logger inside function to make sure all handlers have been added to
    # the root-logger.
    logger = logging.getLogger(__name__)
    for fm_step in ert_config.forward_model_steps:
        logger.info("Config contains forward model step %s", fm_step.name)

    ert = EnKFMain(ert_config)
    if not ert_config.observations and args.mode not in [
        ENSEMBLE_EXPERIMENT_MODE,
        TEST_RUN_MODE,
        WORKFLOW_MODE,
    ]:
        raise ErtCliError(
            f"To run {args.mode}, observations are needed. \n"
            f"Please add an observation file to {args.config}. Example: \n"
            f"'OBS_CONFIG observation_file.txt'."
        )

    if not ert_config.ensemble_config.parameter_configs and args.mode in [
        ENSEMBLE_SMOOTHER_MODE,
        ES_MDA_MODE,
        ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    ]:
        raise ErtCliError(
            f"To run {args.mode}, GEN_KW, FIELD or SURFACE parameters are needed. \n"
            f"Please add to file {args.config}"
        )

    storage = open_storage(ert_config.ens_path, "w")

    if args.mode == WORKFLOW_MODE:
        execute_workflow(ert, storage, args.name)
        return

    status_queue: queue.SimpleQueue[StatusEvents] = queue.SimpleQueue()
    try:
        model = create_model(
            ert_config,
            storage,
            args,
            status_queue,
        )
    except ValueError as e:
        raise ErtCliError(e) from e

    if args.port_range is None and model.queue_system == QueueSystem.LOCAL:
        args.port_range = range(49152, 51819)

    evaluator_server_config = EvaluatorServerConfig(custom_port_range=args.port_range)

    if model.check_if_runpath_exists():
        print(
            "Warning: ERT is running in an existing runpath.\n"
            "Please be aware of the following:\n"
            "- Previously generated results "
            "might be overwritten.\n"
            "- Previously generated files might "
            "be used if not configured correctly.\n"
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
            out = exit_stack.enter_context(
                open(os.devnull, "w", encoding="utf-8")  # noqa
            )
        else:
            out = sys.stderr
        monitor = Monitor(out=out, color_always=args.color_always)
        thread.start()
        try:
            monitor.monitor(status_queue)
        except (SystemExit, KeyboardInterrupt, OSError):
            print("\nKilling simulations...")
            model.cancel()

    thread.join()
    storage.close()

    model.reraise_exception(ErtCliError)
