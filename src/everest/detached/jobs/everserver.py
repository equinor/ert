import argparse
import asyncio
import dataclasses
import datetime
import json
import logging
import logging.config
import os
import queue
import random
import socket
import ssl
import threading
import time
import traceback
import uuid
from base64 import b64decode, b64encode
from contextlib import asynccontextmanager
from functools import lru_cache, partial
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any

import uvicorn
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from dns import resolver, reversename
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketException,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import (
    PlainTextResponse,
    Response,
)
from fastapi.security import (
    HTTPBasic,
    HTTPBasicCredentials,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic import BaseModel

from ert.config import QueueSystem
from ert.ensemble_evaluator import (
    EndEvent,
    EnsembleSnapshot,
    EvaluatorServerConfig,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.run_models import StatusEvents
from ert.run_models.everest_run_model import (
    EverestExitCode,
    EverestRunModel,
)
from ert.trace import tracer
from everest.config import EverestConfig, ServerConfig
from everest.detached import (
    ServerStatus,
    update_everserver_status,
)
from everest.plugins.everest_plugin_manager import EverestPluginManager
from everest.strings import (
    DEFAULT_LOGGING_FORMAT,
    EVEREST,
    EVERSERVER,
    OPT_FAILURE_REALIZATIONS,
    OPTIMIZATION_LOG_DIR,
    OPTIMIZATION_OUTPUT_DIR,
    EverEndpoints,
)
from everest.util import makedirs_if_needed, version_info

logger = logging.getLogger(__name__)


class EverestServerMsg(BaseModel):
    msg: str | None = None


class ServerStarted(EverestServerMsg):
    pass


class ServerStopped(EverestServerMsg):
    pass


class ExperimentComplete(EverestServerMsg):
    exit_code: EverestExitCode
    events: list[StatusEvents]
    server_stopped: bool


class ExperimentFailed(EverestServerMsg):
    pass


@dataclasses.dataclass
class ExperimentRunnerState:
    stop: bool = False
    started: bool = False
    events: list[StatusEvents] = dataclasses.field(default_factory=list)
    subscribers: dict[str, "Subscriber"] = dataclasses.field(default_factory=dict)
    config_path: str | None = None
    start_time_unix: int | None = None


class ExperimentRunner:
    def __init__(
        self,
        everest_config: EverestConfig,
        shared_data: ExperimentRunnerState,
        msg_queue: SimpleQueue[EverestServerMsg],
    ) -> None:
        super().__init__()

        self._everest_config = everest_config
        self._shared_data = shared_data
        self._msg_queue = msg_queue

    async def run(self) -> None:
        status_queue: SimpleQueue[StatusEvents] = SimpleQueue()
        try:
            run_model = EverestRunModel.create(
                self._everest_config,
                optimization_callback=partial(
                    _opt_monitor, shared_data=self._shared_data
                ),
                status_queue=status_queue,
            )

            loop = asyncio.get_running_loop()
            simulation_future = loop.run_in_executor(
                None,
                lambda: run_model.start_simulations_thread(
                    EvaluatorServerConfig()
                    if run_model.queue_config.queue_system == QueueSystem.LOCAL
                    else EvaluatorServerConfig(
                        port_range=(49152, 51819), use_ipc_protocol=False
                    )
                ),
            )
            while True:
                if self._shared_data.stop:
                    run_model.cancel()
                    raise ValueError("Optimization aborted")
                try:
                    item: StatusEvents = status_queue.get(block=False)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                self._shared_data.events.append(item)
                for sub in self._shared_data.subscribers.values():
                    sub.notify()

                if isinstance(item, EndEvent):
                    # Wait for subscribers to receive final events
                    for sub in self._shared_data.subscribers.values():
                        await sub.is_done()

                    break
            await simulation_future
            assert run_model.exit_code is not None
            self._msg_queue.put(
                ExperimentComplete(
                    exit_code=run_model.exit_code,
                    events=self._shared_data.events,
                    server_stopped=self._shared_data.stop,
                )
            )
        except Exception as e:
            logging.getLogger(EVERSERVER).exception(e)
            self._msg_queue.put(
                ExperimentFailed(msg=f"Exception: {e}\n{traceback.format_exc()}")
            )
        finally:
            logging.getLogger(EVERSERVER).info(
                f"ExperimentRunner done. Items left in queue: {status_queue.qsize()}"
            )


class Subscriber:
    """
    This class keeps track of events and allows subscribers
    to wait for new events to occur. Each subscriber instance
    can be notified of an event, at which point any coroutines
    that are waiting for an event will resume execution.
    """

    def __init__(self) -> None:
        self.index = 0
        self._event = asyncio.Event()
        self._done = asyncio.Event()

    def notify(self) -> None:
        self._event.set()

    def done(self):
        self._done.set()

    async def wait_for_event(self) -> None:
        await self._event.wait()
        self._event.clear()

    async def is_done(self) -> None:
        await self._done.wait()


@lru_cache
def _get_machine_name() -> str:
    """Returns a name that can be used to identify this machine in a network

    A fully qualified domain name is returned if available. Otherwise returns
    the string `localhost`
    """
    hostname = socket.gethostname()
    try:
        # We need the ip-address to perform a reverse lookup to deal with
        # differences in how the clusters are getting their fqdn's
        ip_addr = socket.gethostbyname(hostname)
        reverse_name = reversename.from_address(ip_addr)
        resolved_hosts = [
            str(ptr_record).rstrip(".")
            for ptr_record in resolver.resolve(reverse_name, "PTR")
        ]
        resolved_hosts.sort()
        return resolved_hosts[0]
    except (resolver.NXDOMAIN, resolver.NoResolverConfiguration):
        # If local address and reverse lookup not working - fallback
        # to socket fqdn which are using /etc/hosts to retrieve this name
        return socket.getfqdn()
    except socket.gaierror:
        logging.getLogger(EVERSERVER).debug(traceback.format_exc())
        return "localhost"


def _opt_monitor(shared_data: ExperimentRunnerState) -> str | None:
    if shared_data.stop:
        return "stop_optimization"
    return None


def _everserver_thread(
    shared_data: ExperimentRunnerState,
    server_config: dict[str, Any],
    msg_queue: SimpleQueue[EverestServerMsg],
) -> None:
    # ruff: noqa: RUF029
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore
        # Startup event
        msg_queue.put(ServerStarted())
        yield
        # Shutdown event
        msg_queue.put(ServerStopped())

    app = FastAPI(lifespan=lifespan)
    security = HTTPBasic()

    def _check_authentication(auth_header: str) -> None:
        if auth_header is None:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION, reason="No authentication"
            )
        _, encoded_credentials = auth_header.split(" ")
        decoded_credentials = b64decode(encoded_credentials).decode("utf-8")
        _, _, password = decoded_credentials.partition(":")
        if password != server_config["authentication"]:
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    def _check_user(credentials: HTTPBasicCredentials) -> None:
        if credentials.password != server_config["authentication"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )

    def _log(request: Request) -> None:
        logging.getLogger(EVERSERVER).info(
            f"{request.scope['path']} entered from "
            f"{request.client.host if request.client else 'unknown host'} "
            f"with HTTP {request.method}"
        )

    @app.get("/")
    def get_status(
        request: Request, credentials: HTTPBasicCredentials = Depends(security)
    ) -> PlainTextResponse:
        _log(request)
        _check_user(credentials)
        return PlainTextResponse("Everest is running")

    @app.post("/" + EverEndpoints.stop)
    def stop(
        request: Request, credentials: HTTPBasicCredentials = Depends(security)
    ) -> Response:
        _log(request)
        _check_user(credentials)
        shared_data.stop = True
        msg_queue.put(ServerStopped())
        return Response("Raise STOP flag succeeded. Everest initiates shutdown..", 200)

    @app.post("/" + EverEndpoints.start_experiment)
    async def start_experiment(
        request: Request,
        background_tasks: BackgroundTasks,
        credentials: HTTPBasicCredentials = Depends(security),
    ) -> Response:
        _log(request)
        _check_user(credentials)
        if not shared_data.started:
            request_data = await request.json()
            config = EverestConfig.with_plugins(request_data)
            runner = ExperimentRunner(config, shared_data, msg_queue)
            try:
                background_tasks.add_task(runner.run)
                shared_data.started = True

                # Assume only one unique running experiment per everserver instance
                # Ideally, we should return the experiment ID in the response here
                shared_data.config_path = config.config_path

                # Assume client and server is always in the same timezone
                # so disregard timestamps
                shared_data.start_time_unix = int(time.time())
                return Response("Everest experiment started")
            except Exception as e:
                logging.getLogger(EVERSERVER).exception(e)
                return Response(f"Could not start experiment: {e!s}", status_code=501)
        return Response("Everest experiment is running")

    @app.get("/" + EverEndpoints.config_path)
    async def config_path(
        request: Request, credentials: HTTPBasicCredentials = Depends(security)
    ) -> Response:
        _log(request)
        _check_user(credentials)
        if not shared_data.started:
            return Response("No experiment started", status_code=404)

        return Response(str(shared_data.config_path), status_code=200)

    @app.get("/" + EverEndpoints.simulation_dir)
    async def simulation_dir(
        request: Request, credentials: HTTPBasicCredentials = Depends(security)
    ) -> Response:
        _log(request)
        _check_user(credentials)
        if not shared_data.started:
            return Response("No experiment started", status_code=404)

        sim_dir = EverestConfig.from_file(shared_data.config_path).simulation_dir
        return Response(sim_dir, status_code=200)

    @app.get("/" + EverEndpoints.start_time)
    async def start_time(
        request: Request, credentials: HTTPBasicCredentials = Depends(security)
    ) -> Response:
        _log(request)
        _check_user(credentials)
        if not shared_data.started:
            return Response("No experiment started", status_code=404)

        return Response(str(shared_data.start_time_unix), status_code=200)

    @app.websocket("/events")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        _check_authentication(websocket.headers.get("Authorization"))
        subscriber_id = str(uuid.uuid4())
        try:
            while True:
                event = await get_event(subscriber_id=subscriber_id)
                await websocket.send_json(jsonable_encoder(event))
                if isinstance(event, EndEvent):
                    break
        except Exception as e:
            logging.getLogger(EVERSERVER).exception(str(e))
        finally:
            logging.getLogger(EVERSERVER).info(
                f"Subscriber {subscriber_id} done. Closing websocket"
            )
            # Give some time for subscribers to get events
            await asyncio.sleep(5)
            shared_data.subscribers[subscriber_id].done()

    async def get_event(subscriber_id: str) -> StatusEvents:
        """
        The function waits until there is an event available for the subscriber
        and returns the event. If the subscriber is up to date it will
        wait until we wake up the subscriber using notify
        """
        if subscriber_id not in shared_data.subscribers:
            shared_data.subscribers[subscriber_id] = Subscriber()
        subscriber = shared_data.subscribers[subscriber_id]

        while subscriber.index >= len(shared_data.events):
            await subscriber.wait_for_event()

        event = shared_data.events[subscriber.index]
        subscriber.index += 1
        return event

    # Configure the Uvicorn server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=server_config["port"],
        ssl_keyfile=server_config["key_path"],
        ssl_certfile=server_config["cert_path"],
        ssl_version=ssl.PROTOCOL_SSLv23,
        ssl_keyfile_password=server_config["key_passwd"],
        log_level=logging.CRITICAL,
    )


def _find_open_port(host: str, lower: int, upper: int) -> int:
    # Making the port selection random does not fix the problem that an
    # everserver might be assigned a port that another everserver in the process
    # of shutting down already have.
    #
    # Since this problem is very unlikely in the normal usage of everest this change
    # is mainly for alowing testing to run in paralell.

    for _ in range(10):
        port = random.randint(lower, upper)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, port))
            sock.close()
        except OSError:
            logging.getLogger(EVERSERVER).info(f"Port {port} for host {host} is taken")
        else:
            return port
    msg = (
        f"Failed 10 times to get a random port in the range {lower}-{upper} on {host}. "
        "Giving up."
    )
    logging.getLogger(EVERSERVER).error(msg)
    raise Exception(msg)


def _write_hostfile(
    host_file_path: str, host: str, port: int, cert: str, auth: str
) -> None:
    if not os.path.exists(os.path.dirname(host_file_path)):
        os.makedirs(os.path.dirname(host_file_path))
    data = {
        "host": host,
        "port": port,
        "cert": cert,
        "auth": auth,
    }
    json_string = json.dumps(data)

    with open(host_file_path, "w", encoding="utf-8") as f:
        f.write(json_string)


def _configure_loggers(detached_dir: Path, log_dir: Path, logging_level: int) -> None:
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
    optimization_output_dir = str(Path(output_dir).absolute() / OPTIMIZATION_OUTPUT_DIR)

    status_path = ServerConfig.get_everserver_status_path(output_dir)
    host_file = ServerConfig.get_hostfile_path(output_dir)
    msg_queue: SimpleQueue[EverestServerMsg] = SimpleQueue()

    ctx = (
        TraceContextTextMapPropagator().extract(
            carrier={"traceparent": options.traceparent}
        )
        if options.traceparent
        else None
    )

    with tracer.start_as_current_span("everest.everserver", context=ctx):
        try:
            _configure_loggers(
                detached_dir=Path(ServerConfig.get_detached_node_dir(output_dir)),
                log_dir=Path(output_dir) / OPTIMIZATION_LOG_DIR,
                logging_level=options.logging_level,
            )

            logging.getLogger(EVERSERVER).info("Everserver starting ...")
            update_everserver_status(status_path, ServerStatus.starting)
            logger.info(version_info())
            logger.info(f"Output directory: {output_dir}")

            authentication = _generate_authentication()
            cert_path, key_path, key_pw = _generate_certificate(
                ServerConfig.get_certificate_dir(output_dir)
            )
            host = _get_machine_name()
            port = _find_open_port(host, lower=5000, upper=5800)
            _write_hostfile(host_file, host, port, cert_path, authentication)

            shared_data = ExperimentRunnerState()

            server_config = {
                "optimization_output_dir": optimization_output_dir,
                "port": port,
                "cert_path": cert_path,
                "key_path": key_path,
                "key_passwd": key_pw,
                "authentication": authentication,
            }
            # Starting the server
            everserver_instance = threading.Thread(
                target=_everserver_thread,
                args=(shared_data, server_config, msg_queue),
            )
            everserver_instance.daemon = True
            everserver_instance.start()

            # Monitoring the server
            logging.getLogger(EVERSERVER).info("Everserver started")
            while True:
                try:
                    item = msg_queue.get(timeout=1)  # Wait for data
                    match item:
                        case ServerStarted():
                            update_everserver_status(status_path, ServerStatus.running)
                        case ServerStopped():
                            update_everserver_status(status_path, ServerStatus.stopped)
                            return
                        case ExperimentFailed():
                            update_everserver_status(
                                status_path, ServerStatus.failed, item.msg
                            )
                            return
                        case ExperimentComplete():
                            status, message = _get_optimization_status(
                                item.exit_code, item.events, item.server_stopped
                            )
                            update_everserver_status(status_path, status, message)
                            return
                except Empty:
                    continue
        except Exception as e:
            update_everserver_status(
                status_path,
                ServerStatus.failed,
                message=traceback.format_exc(),
            )
            logging.getLogger(EVERSERVER).exception(e)
        finally:
            logging.getLogger(EVERSERVER).info(
                f"Everserver stopped. Items left in queue: {msg_queue.qsize()}"
            )


def _get_optimization_status(
    exit_code: EverestExitCode, events: list[StatusEvents], server_stopped: bool
) -> tuple[ServerStatus, str]:
    match exit_code:
        case EverestExitCode.MAX_BATCH_NUM_REACHED:
            return ServerStatus.completed, "Maximum number of batches reached."

        case EverestExitCode.MAX_FUNCTIONS_REACHED:
            return (
                ServerStatus.completed,
                "Maximum number of function evaluations reached.",
            )

        case EverestExitCode.USER_ABORT:
            return ServerStatus.stopped, "Optimization aborted."

        case EverestExitCode.TOO_FEW_REALIZATIONS:
            status_ = ServerStatus.stopped if server_stopped else ServerStatus.failed
            messages = _failed_realizations_messages(events)
            for msg in messages:
                logging.getLogger(EVEREST).error(msg)
            return status_, "\n".join(messages)
        case _:
            return ServerStatus.completed, "Optimization completed."


def _failed_realizations_messages(events: list[StatusEvents]) -> list[str]:
    snapshots: dict[int, EnsembleSnapshot] = {}
    for event in events:
        if isinstance(event, FullSnapshotEvent) and event.snapshot:
            snapshots[event.iteration] = event.snapshot
        elif isinstance(event, SnapshotUpdateEvent) and event.snapshot:
            snapshot = snapshots[event.iteration]
            assert isinstance(snapshot, EnsembleSnapshot)
            snapshot.merge_snapshot(event.snapshot)
    logging.getLogger("forward_models").info("Status event")
    messages = [OPT_FAILURE_REALIZATIONS]
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
    cert_name = _get_machine_name()
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "NO"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Bergen"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Sandsli"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Equinor"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"{cert_name}"),
        ]
    )
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
            x509.SubjectAlternativeName([x509.DNSName(f"{cert_name}")]),
            critical=False,
        )
        .sign(key, hashes.SHA256(), default_backend())
    )

    # Write certificate and key to disk
    makedirs_if_needed(cert_folder)
    cert_path = os.path.join(cert_folder, cert_name + ".crt")
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    key_path = os.path.join(cert_folder, cert_name + ".key")
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
