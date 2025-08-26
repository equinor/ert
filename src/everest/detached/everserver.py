import argparse
import datetime
import logging
import logging.config
import os
import pathlib
import time
import traceback
from base64 import b64encode
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
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
from ert.shared import get_machine_name
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


def _generate_certificate(cert_folder: str) -> tuple[str, str, bytes]:
    """Generate a private key and a certificate signed with it

    Both the certificate and the key are written to files in the folder given
    by `get_certificate_dir(config)`. The key is encrypted before being
    stored.
    Returns the path to the certificate file, the path to the key file, and
    the password used for encrypting the key
    """
    # Generate private key
    key = rsa.generate_private_key(
        public_exponent=65537, key_size=4096, backend=default_backend()
    )

    # Generate the certificate and sign it with the private key
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "NO"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Bergen"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Sandsli"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Equinor"),
        ]
    )
    dns_name = get_machine_name()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(
            datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=365)
        )  # 1 year
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(f"{dns_name}")]),
            critical=False,
        )
        .sign(key, hashes.SHA256(), default_backend())
    )

    # Write certificate and key to disk
    makedirs_if_needed(cert_folder)
    cert_path = os.path.join(cert_folder, dns_name + ".crt")
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    key_path = os.path.join(cert_folder, dns_name + ".key")
    pw = bytes(os.urandom(28))
    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.BestAvailableEncryption(pw),
            )
        )
    return cert_path, key_path, pw


def _generate_authentication() -> str:
    n_bytes = 128
    random_bytes = bytes(os.urandom(n_bytes))
    return b64encode(random_bytes).decode("utf-8")


if __name__ == "__main__":
    main()
