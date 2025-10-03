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

import httpx
import yaml
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from ert.dark_storage.client import Client as DarkStorageClient
from ert.plugins.plugin_manager import ErtPluginManager
from ert.run_models.run_model import ExperimentStatus
from ert.services import StorageService
from ert.services._base_service import BaseServiceExit
from ert.trace import tracer
from everest.config import ServerConfig
from everest.detached import (
    ExperimentState,
    everserver_status,
    update_everserver_status,
)
from everest.strings import (
    DEFAULT_LOGGING_FORMAT,
    EVEREST,
    EVERSERVER,
    OPTIMIZATION_LOG_DIR,
)
from everest.util import makedirs_if_needed, version_info

logger = logging.getLogger(__name__)


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

    plugin_manager = ErtPluginManager()
    plugin_manager.add_logging_handle_to_root(logging.getLogger())
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

    def _create_client(server_path: str) -> DarkStorageClient:
        return StorageService.session(project=server_path)

    with (
        tracer.start_as_current_span("everest.everserver", context=ctx),
        NamedTemporaryFile() as log_file,
    ):
        client = None
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
                client = _create_client(server_path)
                update_everserver_status(status_path, ExperimentState.running)
                done = False
                while not done:
                    try:
                        response = client.get(
                            "/experiment_server/status", auth=server.fetch_auth()
                        )
                    except httpx.RemoteProtocolError:
                        logger.warning(
                            "httpx.RemoteProtocolError caught when polling "
                            "experiment_server for status. "
                            "Will recreate client and try to continue"
                        )
                        client.close()
                        time.sleep(1)  # Be nice
                        client = _create_client(server_path)
                        continue
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
                            status_path,
                            ExperimentState.failed,
                            message=status.message,
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
        finally:
            if client is not None:
                client.close()


if __name__ == "__main__":
    main()
