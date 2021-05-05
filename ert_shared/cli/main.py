#!/usr/bin/env python
import logging
import os
import sys
import threading

from ert_shared import ERT, clear_global_state
from ert_shared.cli import (
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    WORKFLOW_MODE,
)
from ert_shared.cli.model_factory import create_model
from ert_shared.cli.monitor import Monitor
from ert_shared.cli.notifier import ErtCliNotifier
from ert_shared.cli.workflow import execute_workflow
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.feature_toggling import FeatureToggling
from ert_shared.status.tracker.factory import create_tracker
from res.enkf import EnKFMain, ResConfig


def _clear_and_exit(args):
    clear_global_state()
    sys.exit(args)


def run_cli(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res_config = ResConfig(args.config)
    os.chdir(res_config.config_path)
    ert = EnKFMain(res_config, strict=True, verbose=args.verbose)
    notifier = ErtCliNotifier(ert, args.config)
    ERT.adapt(notifier)

    if args.mode == WORKFLOW_MODE:
        execute_workflow(args.name)
        return

    model, argument = create_model(args)
    # Test run does not have a current_case
    if "current_case" in args and args.current_case:
        ERT.enkf_facade.select_or_create_new_case(args.current_case)

    if (
        args.mode
        in [ENSEMBLE_SMOOTHER_MODE, ITERATIVE_ENSEMBLE_SMOOTHER_MODE, ES_MDA_MODE]
        and args.target_case == ERT.enkf_facade.get_current_case_name()
    ):
        msg = (
            "ERROR: Target file system and source file system can not be the same. "
            "They were both: {}.".format(args.target_case)
        )
        _clear_and_exit(msg)

    ee_config = None
    if FeatureToggling.is_enabled("ensemble-evaluator"):
        ee_config = EvaluatorServerConfig()
        argument.update({"ee_config": ee_config})

    thread = threading.Thread(
        name="ert_cli_simulation_thread",
        target=model.start_simulations_thread,
        args=(argument,),
    )
    thread.start()

    tracker = create_tracker(model, detailed_interval=0, ee_config=ee_config)

    out = open(os.devnull, "w") if args.disable_monitoring else sys.stdout
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
        _clear_and_exit(1)  # the monitor has already reported the error message
