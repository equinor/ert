from __future__ import annotations

import contextlib
import logging
import os
import queue
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from threading import Thread
from typing import Any, TextIO

import httpx

from ert.cli.monitor import Monitor
from ert.cli.workflow import execute_workflow
from ert.config.ert_config import ErtConfig
from ert.ensemble_evaluator.event import EndEvent
from ert.mode_definitions import (
    ENIF_MODE,
    ENSEMBLE_EXPERIMENT_MODE,
    ES_MDA_MODE,
    TEST_RUN_MODE,
    WORKFLOW_MODE,
)
from ert.namespace import Namespace
from ert.plugins import ErtRuntimePlugins, get_site_plugins
from ert.run_models.event import StatusEvents
from ert.services import ErtClient, ErtServerController
from ert.storage import LocalStorage, open_storage
from ert.storage.local_storage import local_storage_set_ert_config
from ert.workflow_runner import WorkflowJobFailedError

_SERVER_RUN_ARGUMENTS = (
    "mode",
    "experiment_name",
    "current_ensemble",
    "target_ensemble",
    "realizations",
    "weights",
    "prior_ensemble_id",
)


class ErtCliError(Exception):
    """Used for exception where the Ert command line interface should
    exit with a nonzero return code
    """


def run_cli(args: Namespace, runtime_plugins: ErtRuntimePlugins | None = None) -> None:
    os.chdir(Path(args.config).resolve().parent)
    # Changing current working directory means we need to update
    # the config file to be the base name of the original config
    args.config = Path(args.config).name

    active_plugins = (
        runtime_plugins if runtime_plugins is not None else get_site_plugins()
    )
    ert_config = ErtConfig.with_plugins(active_plugins).from_file(args.config)

    local_storage_set_ert_config(ert_config)
    counter_fm_steps = Counter(fms.name for fms in ert_config.forward_model_steps)

    # Create logger inside function to make sure all handlers have been added to
    # the root-logger.
    logger = logging.getLogger(__name__)
    for fm_step_name, count in counter_fm_steps.items():
        logger.info(
            f"Config contains forward model step {fm_step_name} {count} time(s)",
        )

    is_esmda_prior_run = args.mode == ES_MDA_MODE and bool(args.prior_ensemble_id)
    if (
        not ert_config.observation_declarations
        and not is_esmda_prior_run
        and args.mode not in {ENSEMBLE_EXPERIMENT_MODE, TEST_RUN_MODE, WORKFLOW_MODE}
    ):
        raise ErtCliError(
            f"To run {args.mode}, observations are needed.\n"
            f"Please add an observation file to {args.config}. Example:\n"
            f"'OBS_CONFIG observation_file.txt'."
        )

    if (
        args.mode == ENIF_MODE
        and ert_config.analysis_config.parameter_type_update_strategies
    ):
        strategies = "".join(
            f"  {param_type.name}: {strategy.name}\n"
            for param_type, strategy in (
                ert_config.analysis_config.parameter_type_update_strategies.items()
            )
        )
        print(
            "Warning: Any given update strategy will be ignored "
            "when running Ensemble Information Filter.\n"
            "The following parameter update strategies were found "
            "in the configuration:\n"
            f"{strategies}"
        )

    if args.mode == WORKFLOW_MODE:
        path = Path(ert_config.ens_path)
        if LocalStorage.check_migration_needed(path):
            LocalStorage.perform_migration(path)
        with open_storage(path, "w") as storage:
            try:
                execute_workflow(ert_config, storage, args.name)
            except WorkflowJobFailedError as err:
                raise ErtCliError(str(err)) from err
        return

    _run_experiment_on_server(ert_config, args)


def _run_experiment_on_server(ert_config: ErtConfig, args: Namespace) -> None:
    storage_path = Path(ert_config.ens_path).absolute()
    if LocalStorage.check_migration_needed(storage_path):
        LocalStorage.perform_migration(storage_path)

    with ErtServerController.init_service(project=storage_path) as server:
        if not server.wait_until_ready():
            raise ErtCliError(f"Failed to start the ERT server for {storage_path}")
        try:
            client = ErtClient.get_client(storage_path)
        except TimeoutError as err:
            raise ErtCliError(f"Failed to connect to the ERT server: {err}") from err

        try:
            config_id = client.register_ert_experiment(
                ert_config, _server_run_arguments(args)
            )
        except httpx.RequestError as err:
            raise ErtCliError(f"Failed to register the experiment: {err}") from err

        started = False
        try:
            _start_registered_experiment(client, config_id, ert_config, args)
            started = True
        except httpx.RequestError as err:
            raise ErtCliError(f"Failed to start the experiment: {err}") from err
        finally:
            if not started:
                with contextlib.suppress(httpx.RequestError):
                    client.discard_registration(config_id)

        event_queue, event_thread = client.setup_event_queue_from_ws_endpoint(config_id)

        def stop_experiment() -> None:
            try:
                client.stop_server()
            except httpx.RequestError as err:
                logging.getLogger(__name__).warning(
                    f"Failed to stop the experiment on the ERT server: {err}"
                )

        _monitor_experiment(
            args, ert_config, event_queue, event_thread, stop_experiment
        )


def _start_registered_experiment(
    client: ErtClient, config_id: str, ert_config: ErtConfig, args: Namespace
) -> None:
    model_data = client.get_runmodel_data(config_id)
    _print_realizations_intersected_with_design_matrix(
        ert_config, args, model_data["number_of_active_realizations"]
    )
    if client.runpath_exists(config_id):
        print(
            "Warning: ERT is running in an existing runpath.\n"
            "Please be aware of the following:\n"
            "- Previously generated results "
            "might be overwritten.\n"
            "- Previously generated files might "
            "be used if not configured correctly.\n"
            f"- {model_data['number_of_existing_runpaths']} out of "
            f"{model_data['number_of_active_realizations']} realizations "
            "are running in existing runpaths.\n"
        )
        logging.getLogger(__name__).warning("ERT is running in an existing runpath")
    client.start_experiment_ert(config_id)


def _server_run_arguments(args: Namespace) -> dict[str, Any]:
    return {
        name: value
        for name in _SERVER_RUN_ARGUMENTS
        if (value := getattr(args, name, None)) is not None
    }


def _print_realizations_intersected_with_design_matrix(
    ert_config: ErtConfig, args: Namespace, number_of_active_realizations: int
) -> None:
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
            f"and DESIGN_MATRIX ({number_of_active_realizations})"
        )


def _monitor_experiment(
    args: Namespace,
    ert_config: ErtConfig,
    event_queue: queue.SimpleQueue[StatusEvents],
    experiment_thread: Thread,
    cancel: Callable[[], None],
) -> None:
    with contextlib.ExitStack() as exit_stack:
        out: TextIO
        if args.disable_monitoring:
            out = exit_stack.enter_context(Path(os.devnull).open("w", encoding="utf-8"))
        else:
            out = sys.stderr
        monitor = Monitor(out=out, color_always=args.color_always)
        experiment_thread.start()
        end_event: EndEvent | None = None
        try:
            end_event = monitor.monitor(
                event_queue, ert_config.analysis_config.log_path
            )
        except Exception:
            print("\nKilling simulations...")
            cancel()

    experiment_thread.join()

    if end_event is not None and end_event.failed:
        # If monitor has not reported, give some info if the job failed
        msg = end_event.msg if args.disable_monitoring else ""
        raise ErtCliError(msg)
