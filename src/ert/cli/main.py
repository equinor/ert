#!/usr/bin/env python
import asyncio
import contextlib
import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import filelock

from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert.cli import ENSEMBLE_EXPERIMENT_MODE, TEST_RUN_MODE, WORKFLOW_MODE
from ert.cli.model_factory import create_model
from ert.cli.monitor import Monitor
from ert.cli.workflow import execute_workflow
from ert.ensemble_evaluator import EvaluatorServerConfig, EvaluatorTracker
from ert.libres_facade import LibresFacade
from ert.shared.feature_toggling import FeatureToggling


class ErtCliError(Exception):
    pass


class ErtTimeoutError(Exception):
    pass


def run_cli(args):
    ert_config = ErtConfig.from_file(args.config)

    # Create logger inside function to make sure all handlers have been added to
    # the root-logger.
    logger = logging.getLogger(__name__)
    for job in ert_config.forward_model_list:
        logger.info("Config contains forward model job %s", job.name)

    for suggestion in ErtConfig.make_suggestion_list(args.config):
        print(f"Warning: {suggestion}")

    os.chdir(ert_config.config_path)
    ert = EnKFMain(ert_config)
    facade = LibresFacade(ert)
    if not facade.have_observations and args.mode not in [
        ENSEMBLE_EXPERIMENT_MODE,
        TEST_RUN_MODE,
        WORKFLOW_MODE,
    ]:
        raise RuntimeError(
            f"To run {args.mode}, observations are needed. \n"
            f"Please add an observation file to {args.config}. Example: \n"
            f"'OBS_CONFIG observation_file.txt'."
        )
    ens_path = Path(ert_config.ens_path)
    storage_lock = filelock.FileLock(ens_path / f"{ens_path.stem}.lock")
    try:
        storage_lock.acquire(timeout=5)
    except filelock.Timeout:
        raise ErtTimeoutError(
            f"Not able to acquire lock for: {ens_path}. You may already be running ert,"
            f" or another user is using the same ENSPATH."
        )
    if args.mode == WORKFLOW_MODE:
        execute_workflow(ert, args.name)
        return

    evaluator_server_config = EvaluatorServerConfig(custom_port_range=args.port_range)
    experiment_id = str(uuid.uuid4())

    # Note that asyncio.run should be called once in ert/shared/main.py
    if FeatureToggling.is_enabled("experiment-server"):
        asyncio.run(
            _run_cli_async(
                ert,
                facade.get_ensemble_size(),
                facade.get_current_case_name(),
                args,
                evaluator_server_config,
                experiment_id,
            ),
            debug=False,
        )
        return
    try:
        model = create_model(
            ert,
            facade.get_ensemble_size(),
            facade.get_current_case_name(),
            args,
            experiment_id,
        )
    except ValueError as e:
        raise ErtCliError(e)

    if model.check_if_runpath_exists():
        print("Warning: ERT is running in an existing runpath")
        logger.warning("ERT is running in an existing runpath")

    # Test run does not have a current_case
    if "current_case" in args and args.current_case:
        facade.select_or_create_new_case(args.current_case)

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
        if args.disable_monitoring:
            out = exit_stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
        else:
            out = sys.stderr
        monitor = Monitor(out=out, color_always=args.color_always)

        try:
            monitor.monitor(tracker.track())
        except (SystemExit, KeyboardInterrupt):
            print("\nKilling simulations...")
            tracker.request_termination()

    thread.join()

    if storage_lock.is_locked:
        storage_lock.release()
        os.remove(storage_lock.lock_file)
    if model.hasRunFailed():
        raise ErtCliError(model.getFailMessage())


# pylint: disable=too-many-arguments
async def _run_cli_async(
    ert: EnKFMain,
    ensemble_size: int,
    current_case_name: str,
    args: Any,
    ee_config: EvaluatorServerConfig,
    experiment_id: str,
):
    # pylint: disable=import-outside-toplevel
    from ert.experiment_server import ExperimentServer

    experiment_server = ExperimentServer(ee_config)
    experiment_server.add_experiment(
        create_model(ert, ensemble_size, current_case_name, args, experiment_id)
    )
    await experiment_server.run_experiment(experiment_id=experiment_id)
