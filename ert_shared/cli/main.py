#!/usr/bin/env python
import logging
import os
import sys
import threading

from ert_shared import ERT
from ert_shared import clear_global_state
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.monitor import create as create_ee_monitor
from ert_shared.feature_toggling import FeatureToggling
from ert_shared.cli.model_factory import create_model
from ert_shared.cli.monitor import Monitor
from ert_shared.cli.notifier import ErtCliNotifier
from ert_shared.cli.workflow import execute_workflow
from ert_shared.cli import WORKFLOW_MODE, ITERATIVE_ENSEMBLE_SMOOTHER_MODE, ENSEMBLE_SMOOTHER_MODE, ES_MDA_MODE, ENSEMBLE_EXPERIMENT_MODE
from ert_shared.tracker.factory import create_tracker
from res.enkf import EnKFMain, ResConfig


def _clear_and_exit(args):
    clear_global_state()
    sys.exit(args)


def run_cli(args):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if FeatureToggling.is_enabled("prefect"):
        ert_conf_path = os.path.abspath(args.config)[:-3] + "ert"
        with open(ert_conf_path, "w") as f:
            f.write("""QUEUE_SYSTEM LOCAL
QUEUE_OPTION LOCAL MAX_RUNNING 50
RUNPATH out/real_%d/iter_%d
NUM_REALIZATIONS 100
MIN_REALIZATIONS 1
""")
        res_config = ResConfig(ert_conf_path)
    else:
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

    if (args.mode in [ENSEMBLE_SMOOTHER_MODE, ITERATIVE_ENSEMBLE_SMOOTHER_MODE, ES_MDA_MODE] and 
        args.target_case == ERT.enkf_facade.get_current_case_name()):
        msg = (
            "ERROR: Target file system and source file system can not be the same. "
            "They were both: {}.".format(args.target_case)
        )
        _clear_and_exit(msg)

    thread = threading.Thread(
        name="ert_cli_simulation_thread",
        target=model.startSimulations,
        args=(argument,)
    )
    thread.start()

    tracker = create_tracker(model, detailed_interval=0)

    out = open(os.devnull, 'w') if args.disable_monitoring else sys.stdout
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

