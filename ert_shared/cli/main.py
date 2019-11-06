#!/usr/bin/env python
import logging
import os
import sys
import threading

from ert_shared import ERT
from ert_shared.cli.model_factory import create_model
from ert_shared.cli.monitor import Monitor
from ert_shared.cli.notifier import ErtCliNotifier
from ert_shared.cli.workflow import execute_workflow
from ert_shared.models import SimulationsTracker
from res.enkf import EnKFMain, ResConfig


def run_cli(args):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    res_config = ResConfig(args.config)
    os.chdir(res_config.config_path)
    ert = EnKFMain(res_config, strict=True, verbose=args.verbose)
    notifier = ErtCliNotifier(ert, args.config)
    ERT.adapt(notifier)

    if args.mode == 'workflow':
        execute_workflow(args.name)
        return

    model, argument = create_model(args)

    if args.disable_monitoring:
        model.startSimulations(argument)
        if model.hasRunFailed():
            sys.exit(model.getFailMessage())
    else:
        thread = threading.Thread(
            name="ert_cli_simulation_thread",
            target=model.startSimulations,
            args=(argument,)
        )
        thread.start()

        tracker = SimulationsTracker(model=model)
        monitor = Monitor(color_always=args.color_always)

        try:
            monitor.monitor(tracker)
        except (SystemExit, KeyboardInterrupt):
            print("\nKilling simulations...")
            model.killAllSimulations()

        thread.join()

        if model.hasRunFailed():
            sys.exit(1)  # the monitor has already reported the error message
