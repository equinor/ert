#!/usr/bin/env python
import logging
import os
import sys
import threading
from ert.ensemble_evaluator import EvaluatorTracker

from ert_shared.cli import (
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    WORKFLOW_MODE,
)
from ert_shared.cli.model_factory import create_model
from ert_shared.cli.monitor import Monitor
from ert_shared.cli.workflow import execute_workflow
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.libres_facade import LibresFacade
from res.enkf import EnKFMain, ResConfig


class ErtCliError(Exception):
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
    ert = EnKFMain(res_config, strict=True, verbose=args.verbose)
    facade = LibresFacade(ert)

    if args.mode == WORKFLOW_MODE:
        execute_workflow(ert, args.name)
        return
    model = create_model(
        ert,
        facade.get_ensemble_size(),
        facade.get_current_case_name(),
        args,
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
            "They were both: {}.".format(args.target_case)
        )
        raise ErtCliError(msg)

    evaluator_server_config = EvaluatorServerConfig(custom_port_range=args.port_range)

    thread = threading.Thread(
        name="ert_cli_simulation_thread",
        target=model.start_simulations_thread,
        args=(evaluator_server_config,),
    )
    thread.start()

    tracker = EvaluatorTracker(
        model, ee_con_info=evaluator_server_config.get_connection_info()
    )

    out = open(os.devnull, "w") if args.disable_monitoring else sys.stderr
    monitor = Monitor(out=out, color_always=args.color_always)

    try:
        monitor.monitor(tracker)
    except (SystemExit, KeyboardInterrupt):
        print("\nKilling simulations...")
        tracker.request_termination()

    if args.disable_monitoring:
        out.close()

    thread.join()

    if model.hasRunFailed():
        raise ErtCliError(model.getFailMessage())
