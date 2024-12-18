import argparse
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import requests
from ropt.enums import OptimizerExitCode

from everest import export_to_csv, export_with_progress
from everest.config import EverestConfig, ServerConfig
from everest.detached import (
    PROXY,
    ServerStatus,
    update_everserver_status,
)
from everest.detached.jobs.everest_server_api import (
    EverestServerAPI,
    ExperimentRunnerStatus,
)
from everest.export import check_for_errors
from everest.simulator import JOB_FAILURE
from everest.strings import (
    DEFAULT_LOGGING_FORMAT,
    EVEREST,
    OPT_FAILURE_REALIZATIONS,
    SHARED_DATA_ENDPOINT,
    SIM_PROGRESS_ENDPOINT,
    STOP_ENDPOINT,
)
from everest.util import get_azure_logging_handler, makedirs_if_needed, version_info


def _get_optimization_status(exit_code, shared_data):
    if exit_code == "max_batch_num_reached":
        return ServerStatus.completed, "Maximum number of batches reached."

    if exit_code == OptimizerExitCode.MAX_FUNCTIONS_REACHED:
        return ServerStatus.completed, "Maximum number of function evaluations reached."

    if exit_code == OptimizerExitCode.USER_ABORT:
        return ServerStatus.stopped, "Optimization aborted."

    if exit_code == OptimizerExitCode.TOO_FEW_REALIZATIONS:
        status = (
            ServerStatus.stopped if shared_data[STOP_ENDPOINT] else ServerStatus.failed
        )
        messages = _failed_realizations_messages(shared_data)
        for msg in messages:
            logging.getLogger(EVEREST).error(msg)
        return status, "\n".join(messages)

    return ServerStatus.completed, "Optimization completed."


def _failed_realizations_messages(shared_data):
    messages = [OPT_FAILURE_REALIZATIONS]
    failed = shared_data[SIM_PROGRESS_ENDPOINT]["status"]["failed"]
    if failed > 0:
        # Find the set of jobs that failed. To keep the order in which they
        # are found in the queue, use a dict as sets are not ordered.
        failed_jobs = dict.fromkeys(
            job["name"]
            for queue in shared_data[SIM_PROGRESS_ENDPOINT]["progress"]
            for job in queue
            if job["status"] == JOB_FAILURE
        ).keys()
        messages.append(
            "{} job failures caused by: {}".format(failed, ", ".join(failed_jobs))
        )
    return messages


def _configure_loggers(detached_dir: Path, log_dir: Path, logging_level: int) -> None:
    def make_handler_config(
        path: Path, log_level: str | int = "INFO"
    ) -> dict[str, Any]:
        makedirs_if_needed(path.parent)
        return {
            "class": "logging.FileHandler",
            "formatter": "default",
            "level": log_level,
            "filename": path,
        }

    def azure_handler():
        azure_handler = get_azure_logging_handler()
        if azure_handler:
            return azure_handler
        return logging.NullHandler()

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {
            "root": {"level": "NOTSET", "class": "logging.NullHandler"},
            "res": make_handler_config(detached_dir / "simulations.log"),
            "everserver": make_handler_config(detached_dir / "endpoint.log"),
            "everest": make_handler_config(log_dir / "everest.log", logging_level),
            "forward_models": make_handler_config(
                log_dir / "forward_models.log", logging_level
            ),
            "azure_handler": {"()": azure_handler},
        },
        "loggers": {
            "": {"handlers": ["root"], "level": "NOTSET"},
            "res": {"handlers": ["res"]},
            "everserver": {"handlers": ["everserver"]},
            "everest": {"handlers": ["everest", "azure_handler"]},
            "forward_models": {"handlers": ["forward_models"]},
        },
        "formatters": {
            "default": {"format": DEFAULT_LOGGING_FORMAT},
        },
    }

    logging.config.dictConfig(logging_config)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config-file", type=str)
    arg_parser.add_argument("--debug", action="store_true")
    options = arg_parser.parse_args()
    config = EverestConfig.load_file(options.config_file)
    if options.debug:
        config.logging_level = "debug"
    status_path = ServerConfig.get_everserver_status_path(config.output_dir)

    try:
        _configure_loggers(
            detached_dir=Path(ServerConfig.get_detached_node_dir(config.output_dir)),
            log_dir=(
                Path(config.output_dir) / "logs"
                if config.log_dir is None
                else Path(config.log_dir)
            ),
            logging_level=config.logging_level,
        )

        update_everserver_status(status_path, ServerStatus.starting)
        logging.getLogger(EVEREST).info(version_info())
        logging.getLogger(EVEREST).info(f"Output directory: {config.output_dir}")
        logging.getLogger(EVEREST).debug(str(options))

        shared_data = {
            SIM_PROGRESS_ENDPOINT: {},
            STOP_ENDPOINT: False,
        }

        everest_server_api = EverestServerAPI(
            output_dir=config.output_dir,
            optimization_output_dir=config.optimization_output_dir,
        )
        everest_server_api.daemon = True
        everest_server_api.start()

        server_context = (ServerConfig.get_server_context(config.output_dir),)
        url, cert, auth = server_context[0]

    except:
        update_everserver_status(
            status_path,
            ServerStatus.failed,
            message=traceback.format_exc(),
        )
        return

    try:
        # wait until the api server is running
        is_running = False
        while not is_running:
            try:
                requests.get(url + "/", verify=cert, auth=auth, proxies=PROXY)  # type: ignore
                is_running = True
            except:
                time.sleep(1)

        update_everserver_status(status_path, ServerStatus.running)

        is_done = False
        while not is_done:
            resp: requests.Response = requests.get(
                url + "/",
                verify=cert,
                auth=auth,
                proxies=PROXY,  # type: ignore
            )
            server_status = ExperimentRunnerStatus.model_validate_json(
                resp.text if hasattr(resp, "text") else resp.body
            )
            if (
                server_status.message
                and "Everest server is running" in server_status.message
            ):
                time.sleep(1)
            else:
                is_done = True

        if server_status.message:
            update_everserver_status(
                status_path,
                ServerStatus.failed,
                message=server_status.message,
            )
            return

        response: requests.Response = requests.get(
            url + "/" + SHARED_DATA_ENDPOINT,
            verify=cert,
            auth=auth,
            proxies=PROXY,  # type: ignore
        )
        if json_body := json.loads(
            response.text if hasattr(response, "text") else response.body
        ):
            shared_data = json_body

        status, message = _get_optimization_status(server_status.exit_code, shared_data)
        if status != ServerStatus.completed:
            update_everserver_status(status_path, status, message)
            return

    except:
        if shared_data[STOP_ENDPOINT]:
            update_everserver_status(
                status_path,
                ServerStatus.stopped,
                message="Optimization aborted.",
            )
        else:
            update_everserver_status(
                status_path,
                ServerStatus.failed,
                message=traceback.format_exc(),
            )
        return

    try:
        # Exporting data
        update_everserver_status(status_path, ServerStatus.exporting_to_csv)

        if config.export is not None:
            err_msgs, export_ecl = check_for_errors(
                config=config.export,
                optimization_output_path=config.optimization_output_dir,
                storage_path=config.storage_dir,
                data_file_path=config.model.data_file,
            )
            for msg in err_msgs:
                logging.getLogger(EVEREST).warning(msg)
        else:
            export_ecl = True

        export_to_csv(
            data_frame=export_with_progress(config, export_ecl),
            export_path=config.export_path,
        )
    except:
        update_everserver_status(
            status_path,
            ServerStatus.failed,
            message=traceback.format_exc(),
        )
        return

    update_everserver_status(status_path, ServerStatus.completed, message=message)
