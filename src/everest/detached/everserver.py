import argparse
import logging
import logging.config
import os
import pathlib
import time
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic import BaseModel

from ert.ensemble_evaluator import (
    EnsembleSnapshot,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.run_models import StatusEvents
from ert.run_models.everest_run_model import (
    EverestExitCode,
)
from ert.services import StorageService
from ert.services._base_service import BaseServiceExit
from ert.trace import tracer
from everest.config import ServerConfig
from everest.detached import (
    ExperimentState,
    everserver_status,
    update_everserver_status,
)
from everest.plugins.everest_plugin_manager import EverestPluginManager
from everest.strings import (
    DEFAULT_LOGGING_FORMAT,
    EVEREST,
    EVERSERVER,
    OPT_FAILURE_ALL_REALIZATIONS,
    OPT_FAILURE_REALIZATIONS,
    OPTIMIZATION_LOG_DIR,
)
from everest.util import makedirs_if_needed, version_info

logger = logging.getLogger(__name__)


class ExperimentStatus(BaseModel):
    message: str = ""
    status: ExperimentState = ExperimentState.pending


def _configure_loggers(
    detached_dir: Path, log_dir: Path, logging_level: int, output_file: str | None
) -> None:
    def make_handler_config(path: Path, log_level: int) -> dict[str, Any]:
        makedirs_if_needed(str(path.parent))
        return {
            "class": "logging.FileHandler",
            "formatter": "default",
            "level": log_level,
            "filename": path,
        }

    logging_config = {
        "version": 1,
        "handlers": {
            "endpoint_log": make_handler_config(
                detached_dir / "endpoint.log", logging_level
            ),
            "everest_log": make_handler_config(log_dir / "everest.log", logging_level),
            "forward_models_log": make_handler_config(
                log_dir / "forward_models.log", logging_level
            ),
        },
        "loggers": {
            "root": {"handlers": ["endpoint_log"], "level": logging_level},
            EVERSERVER: {
                "handlers": ["endpoint_log"],
                "level": logging_level,
                "propagate": False,
            },
            EVEREST: {
                "handlers": ["everest_log"],
                "level": logging_level,
                "propagate": False,
            },
            "forward_models": {
                "handlers": ["forward_models_log"],
                "level": logging_level,
                "propagate": False,
            },
            "ert.scheduler.job": {
                "handlers": ["forward_models_log"],
                "propagate": False,
                "level": logging_level,
            },
        },
        "formatters": {
            "default": {"format": DEFAULT_LOGGING_FORMAT},
        },
    }

    def path_representer(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))

    yaml.add_representer(pathlib.PosixPath, path_representer)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as outfile:
            yaml.dump(logging_config, outfile, default_flow_style=False)
    logging.config.dictConfig(logging_config)

    plugin_manager = EverestPluginManager()
    plugin_manager.add_log_handle_to_root()
    plugin_manager.add_span_processor_to_trace_provider()


def get_trace_context():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--traceparent",
        type=str,
        help="Trace parent id to be used by the storage root span",
        default=None,
    )
    options = arg_parser.parse_args()
    ctx = (
        TraceContextTextMapPropagator().extract(
            carrier={"traceparent": options.traceparent}
        )
        if options.traceparent
        else None
    )
    return ctx


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--output-dir", "-o", type=str)
    arg_parser.add_argument("--logging-level", "-l", type=int, default=logging.INFO)
    arg_parser.add_argument(
        "--traceparent",
        type=str,
        help="Trace parent id to be used by the storage root span",
        default=None,
    )
    options = arg_parser.parse_args()

    output_dir = options.output_dir

    status_path = ServerConfig.get_everserver_status_path(output_dir)

    ctx = (
        TraceContextTextMapPropagator().extract(
            carrier={"traceparent": options.traceparent}
        )
        if options.traceparent
        else None
    )

    with (
        tracer.start_as_current_span("everest.everserver", context=ctx),
        NamedTemporaryFile() as log_file,
    ):
        try:
            _configure_loggers(
                detached_dir=Path(ServerConfig.get_detached_node_dir(output_dir)),
                log_dir=Path(output_dir) / OPTIMIZATION_LOG_DIR,
                logging_level=options.logging_level,
                output_file=log_file.name,
            )

            logging.getLogger(EVERSERVER).info("Everserver starting ...")
            update_everserver_status(status_path, ExperimentState.pending)
            logger.info(version_info())
            logger.info(f"Output directory: {output_dir}")
            # Starting the server
            server_path = os.path.abspath(ServerConfig.get_session_dir(output_dir))
            status = ""
            with StorageService.init_service(
                timeout=240, project=server_path, logging_config=log_file.name
            ) as server:
                server.fetch_conn_info()
                with StorageService.session(project=server_path) as client:
                    update_everserver_status(status_path, ExperimentState.running)
                    done = False
                    while not done:
                        response = client.get(
                            "/experiment_server/status", auth=server.fetch_auth()
                        )
                        status = ExperimentStatus(**response.json())
                        done = status.status not in {
                            ExperimentState.pending,
                            ExperimentState.running,
                        }
                        time.sleep(0.5)
                    if status.status == ExperimentState.completed:
                        update_everserver_status(
                            status_path,
                            ExperimentState.completed,
                            message=status.message,
                        )
                    elif status.status == ExperimentState.stopped:
                        update_everserver_status(
                            status_path,
                            ExperimentState.stopped,
                            message=status.message,
                        )
                    elif status.status == ExperimentState.failed:
                        update_everserver_status(
                            status_path, ExperimentState.failed, message=status.message
                        )
        except BaseServiceExit:
            # Server exit, happens on normal shutdown and keyboard interrupt
            server_status = everserver_status(status_path)
            if server_status["status"] == ExperimentState.running:
                update_everserver_status(status_path, ExperimentState.stopped)
        except Exception as e:
            update_everserver_status(
                status_path,
                ExperimentState.failed,
                message=traceback.format_exc(),
            )
            logging.getLogger(EVERSERVER).exception(e)


def _get_optimization_status(
    exit_code: EverestExitCode, events: list[StatusEvents]
) -> tuple[ExperimentState, str]:
    match exit_code:
        case EverestExitCode.MAX_BATCH_NUM_REACHED:
            return ExperimentState.completed, "Maximum number of batches reached."

        case EverestExitCode.MAX_FUNCTIONS_REACHED:
            return (
                ExperimentState.completed,
                "Maximum number of function evaluations reached.",
            )

        case EverestExitCode.USER_ABORT:
            return ExperimentState.stopped, "Optimization aborted."

        case (
            EverestExitCode.TOO_FEW_REALIZATIONS
            | EverestExitCode.ALL_REALIZATIONS_FAILED
        ):
            status_ = ExperimentState.failed
            messages = _failed_realizations_messages(events, exit_code)
            for msg in messages:
                logging.getLogger(EVEREST).error(msg)
            return status_, "\n".join(messages)
        case _:
            return ExperimentState.completed, "Optimization completed."


def _failed_realizations_messages(
    events: list[StatusEvents], exit_code: EverestExitCode
) -> list[str]:
    snapshots: dict[int, EnsembleSnapshot] = {}
    for event in events:
        if isinstance(event, FullSnapshotEvent) and event.snapshot:
            snapshots[event.iteration] = event.snapshot
        elif isinstance(event, SnapshotUpdateEvent) and event.snapshot:
            snapshot = snapshots[event.iteration]
            assert isinstance(snapshot, EnsembleSnapshot)
            snapshot.merge_snapshot(event.snapshot)
    logging.getLogger("forward_models").info("Status event")
    messages = [
        OPT_FAILURE_REALIZATIONS
        if exit_code == EverestExitCode.TOO_FEW_REALIZATIONS
        else OPT_FAILURE_ALL_REALIZATIONS
    ]
    for snapshot in snapshots.values():
        for job in snapshot.get_all_fm_steps().values():
            if error := job.get("error"):
                msg = f"{job['name']} Failed with: {error}"
                if msg not in messages:
                    messages.append(msg)
    return messages


if __name__ == "__main__":
    main()
