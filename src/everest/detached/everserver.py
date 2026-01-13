import argparse
import logging
import logging.config
import os
import pathlib
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from ert.plugins.plugin_manager import ErtPluginManager
from ert.services import ErtServer
from ert.services._base_service import BaseServiceExit
from ert.storage import ExperimentStatus
from ert.storage.local_experiment import ExperimentState
from ert.trace import tracer
from ert.utils import makedirs_if_needed
from everest.config import ServerConfig
from everest.strings import (
    DEFAULT_LOGGING_FORMAT,
    EVEREST,
    EVERSERVER,
    EXPERIMENT_SERVER,
    OPTIMIZATION_LOG_DIR,
)
from everest.util import version_info

logger = logging.getLogger(__name__)


def _configure_loggers(
    log_dir: Path, logging_level: int, output_file: str | None
) -> None:
    def make_handler_config(path: Path, log_level: int) -> dict[str, Any]:
        makedirs_if_needed(path.parent)
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
                log_dir / "everserver.log", logging_level
            ),
            "everest_log": make_handler_config(log_dir / "everest.log", logging_level),
            "forward_models_log": make_handler_config(
                log_dir / "forward_models.log", logging_level
            ),
        },
        "loggers": {
            "root": {"handlers": ["endpoint_log"], "level": logging_level},
            "uvicorn": {
                "level": logging.WARNING,
            },
            EVERSERVER: {
                "handlers": ["endpoint_log"],
                "level": logging_level,
                "propagate": False,
            },
            EXPERIMENT_SERVER: {
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
        Path(output_file).write_text(
            yaml.dump(logging_config, default_flow_style=False), encoding="utf-8"
        )
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
                log_dir=Path(output_dir) / OPTIMIZATION_LOG_DIR,
                logging_level=options.logging_level,
                output_file=log_file.name,
            )

            logging.getLogger(EVERSERVER).info("Everserver starting ...")
            logger.info(version_info())
            logger.info(f"Output directory: {output_dir}")
            # Starting the server
            server_path = os.path.abspath(ServerConfig.get_session_dir(output_dir))
            status = ""
            with ErtServer.init_service(
                timeout=240, project=Path(server_path), logging_config=log_file.name
            ) as server:
                server.fetch_connection_info()
                with ErtServer.session(project=Path(server_path)) as client:
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
        except BaseServiceExit:
            # Server exit, happens on normal shutdown and keyboard interrupt
            logging.getLogger(EVERSERVER).info("Everserver stopped by user")
        except Exception as e:
            logging.getLogger(EVERSERVER).exception(e)


if __name__ == "__main__":
    main()
