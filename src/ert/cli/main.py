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

from ert._c_wrappers.enkf import EnKFMain, ResConfig
from ert.cli import (
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    WORKFLOW_MODE,
)
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
    res_config = ResConfig(args.config)

    # Create logger inside function to make sure all handlers have been added to
    # the root-logger.
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging forward model jobs",
        extra={
            "workflow_jobs": str(res_config.model_config.getForwardModel().joblist())
        },
    )

    os.chdir(res_config.config_path)
    ert = EnKFMain(res_config)
    facade = LibresFacade(ert)
    ens_path = Path(res_config.model_config.getEnspath())
    storage_lock = filelock.FileLock(ens_path / (ens_path.stem + ".lock"))
    try:
        storage_lock.acquire(timeout=5)
    except filelock.Timeout:
        raise ErtTimeoutError(
            f"Not able to acquire lock for: {ens_path}, ert could be opened twice, or "
            f"another user is using the same ENSPATH"
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

    model = create_model(
        ert,
        facade.get_ensemble_size(),
        facade.get_current_case_name(),
        args,
        experiment_id,
    )
    # Test run does not have a current_case
    if "current_case" in args and args.current_case:
        facade.select_or_create_new_case(args.current_case)

    if (
        args.mode
        in [ENSEMBLE_SMOOTHER_MODE, ITERATIVE_ENSEMBLE_SMOOTHER_MODE, ES_MDA_MODE]
        and args.target_case == facade.get_current_case_name()
    ):
        msg = (
            "ERROR: Target file system and source file system can not be the same. "
            f"They were both: {args.target_case}."
        )
        raise ErtCliError(msg)

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


async def _run_cli_async(
    ert: EnKFMain,
    ensemble_size: int,
    current_case_name: str,
    args: Any,
    ee_config: EvaluatorServerConfig,
    experiment_id: str,
):
    from ert.experiment_server import ExperimentServer  # noqa

    experiment_server = ExperimentServer(ee_config)
    experiment_server.add_experiment(
        create_model(ert, ensemble_size, current_case_name, args, experiment_id)
    )
    await experiment_server.run_experiment(experiment_id=experiment_id)
