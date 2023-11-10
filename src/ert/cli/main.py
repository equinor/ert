#!/usr/bin/env python
import contextlib
import logging
import os
import sys
import threading
from typing import Any, TextIO

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
from ert.ensemble_evaluator import EvaluatorServerConfig, EvaluatorTracker
from ert.libres_facade import LibresFacade
from ert.namespace import Namespace
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
    for job in ert_config.forward_model_list:
        logger.info("Config contains forward model job %s", job.name)

    ert = EnKFMain(ert_config)
    facade = LibresFacade(ert)
    if not facade.have_observations and args.mode not in [
        ENSEMBLE_EXPERIMENT_MODE,
        TEST_RUN_MODE,
        WORKFLOW_MODE,
    ]:
        raise ErtCliError(
            f"To run {args.mode}, observations are needed. \n"
            f"Please add an observation file to {args.config}. Example: \n"
            f"'OBS_CONFIG observation_file.txt'."
        )

    if not facade.have_smoother_parameters and args.mode in [
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

    experiment = storage.create_experiment(
        parameters=ert_config.ensemble_config.parameter_configuration,
        responses=ert_config.ensemble_config.response_configuration,
        observations=ert_config.observations,
    )

    try:
        model = create_model(
            ert_config,
            storage,
            args,
            experiment.id,
        )
    except ValueError as e:
        raise ErtCliError(e) from e

    if args.port_range is None and model.queue_system == QueueSystem.LOCAL:
        args.port_range = range(49152, 51819)

    evaluator_server_config = EvaluatorServerConfig(custom_port_range=args.port_range)

    experiment.write_simulation_arguments(model.simulation_arguments)

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

    thread = threading.Thread(
        name="ert_cli_simulation_thread",
        target=model.start_simulations_thread,
        args=(evaluator_server_config,),
    )
    thread.start()

    tracker = EvaluatorTracker(
        model, ee_con_info=evaluator_server_config.get_connection_info()
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

        try:
            monitor.monitor(tracker.track())
        except (SystemExit, KeyboardInterrupt):
            print("\nKilling simulations...")
            tracker.request_termination()

    thread.join()
    storage.close()

    if model.hasRunFailed():
        raise ErtCliError(model.getFailMessage())
