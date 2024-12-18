import json
import logging
import os
import socket
import ssl
import threading
import traceback
from base64 import b64encode
from datetime import datetime, timedelta
from functools import partial

import uvicorn
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import (
    JSONResponse,
    Response,
)
from fastapi.security import (
    HTTPBasic,
    HTTPBasicCredentials,
)
from pydantic import BaseModel
from ropt.enums import OptimizerExitCode

from ert.config import QueueSystem
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel, EverestRunModelExitCode
from ert.shared import get_machine_name as ert_shared_get_machine_name
from everest.config import EverestConfig, ServerConfig
from everest.detached import get_opt_status
from everest.strings import (
    OPT_PROGRESS_ENDPOINT,
    SHARED_DATA_ENDPOINT,
    SIM_PROGRESS_ENDPOINT,
    START_ENDPOINT,
    STOP_ENDPOINT,
)
from everest.util import makedirs_if_needed


def _find_open_port(host: str, lower: int, upper: int) -> int:
    for port in range(lower, upper):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, port))
            sock.close()
            return port
        except OSError:
            logging.getLogger("everserver").info(
                "Port {} for host {} is taken".format(port, host)
            )
    msg = "No open port for host {} in the range {}-{}".format(host, lower, upper)
    logging.getLogger("everserver").exception(msg)
    raise Exception(msg)


def _write_hostfile(host_file_path, host, port, cert, auth) -> None:
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


def _generate_authentication() -> str:
    n_bytes = 128
    random_bytes = bytes(os.urandom(n_bytes))
    return b64encode(random_bytes).decode("utf-8")


def _generate_certificate(cert_folder: str):
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
    cert_name = ert_shared_get_machine_name()
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "NO"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Bergen"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Sandsli"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Equinor"),
            x509.NameAttribute(NameOID.COMMON_NAME, "{}".format(cert_name)),
        ]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(datetime.utcnow() + timedelta(days=365))  # 1 year
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("{}".format(cert_name))]),
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


def _sim_monitor(context_status, event=None, shared_data=None):
    status = context_status["status"]
    assert shared_data
    shared_data[SIM_PROGRESS_ENDPOINT] = {
        "batch_number": context_status["batch_number"],
        "status": {
            "running": status.get("Running", 0),
            "waiting": status.get("Waiting", 0),
            "pending": status.get("Pending", 0),
            "complete": status.get("Finished", 0),
            "failed": status.get("Failed", 0),
        },
        "progress": context_status["progress"],
    }

    if shared_data[STOP_ENDPOINT]:
        return "stop_queue"


def _opt_monitor(shared_data=None):
    assert shared_data
    if shared_data[STOP_ENDPOINT]:
        return "stop_optimization"


class ExperimentRunnerStatus(BaseModel):
    status: str | None = None
    exit_code: EverestRunModelExitCode | OptimizerExitCode | None = None
    message: str | None = None


class ExperimentRunner(threading.Thread):
    def __init__(self, everest_config, state: dict):
        super().__init__()

        self.everest_config = everest_config
        self.state = state
        self.status: ExperimentRunnerStatus | None = None

    def run(self):
        run_model = EverestRunModel.create(
            self.everest_config,
            simulation_callback=partial(_sim_monitor, shared_data=self.state),
            optimization_callback=partial(_opt_monitor, shared_data=self.state),
        )

        evaluator_server_config = EvaluatorServerConfig(
            custom_port_range=range(49152, 51819)
            if run_model.ert_config.queue_config.queue_system == QueueSystem.LOCAL
            else None
        )

        try:
            run_model.run_experiment(evaluator_server_config)
            self.status = ExperimentRunnerStatus(
                status="Experiment finished", exit_code=run_model.exit_code
            )
        except Exception:
            self.status = ExperimentRunnerStatus(
                status="Experiment failed", message=traceback.format_exc()
            )

    def get_status(self) -> ExperimentRunnerStatus | None:
        return self.status


security = HTTPBasic()


class EverestServerAPI(threading.Thread):
    def __init__(self, output_dir: str, optimization_output_dir: str):
        super().__init__()

        self.output_dir = output_dir
        self.optimization_output_dir = optimization_output_dir

        self.app = FastAPI()

        self.router = APIRouter()
        self.router.add_api_route("/", self.get_status, methods=["GET"])
        self.router.add_api_route("/" + STOP_ENDPOINT, self.stop, methods=["POST"])
        self.router.add_api_route(
            "/" + SIM_PROGRESS_ENDPOINT, self.get_sim_progress, methods=["GET"]
        )
        self.router.add_api_route(
            "/" + OPT_PROGRESS_ENDPOINT, self.get_opt_progress, methods=["GET"]
        )
        self.router.add_api_route(
            "/" + START_ENDPOINT, self.start_experiment, methods=["POST"]
        )
        self.router.add_api_route(
            "/" + SHARED_DATA_ENDPOINT, self.get_state, methods=["GET"]
        )

        self.app.include_router(self.router)

        self.state = {
            SIM_PROGRESS_ENDPOINT: {},
            STOP_ENDPOINT: False,
        }

        self.runner: ExperimentRunner | None = None

        # same code is in ensemble evaluator
        self.authentication = _generate_authentication()

        # same code is in ensemble evaluator
        self.cert_path, self.key_path, self.key_pw = _generate_certificate(
            ServerConfig.get_certificate_dir(self.output_dir)
        )
        self.host = ert_shared_get_machine_name()
        self.port = _find_open_port(self.host, lower=5000, upper=5800)

        host_file = ServerConfig.get_hostfile_path(self.output_dir)
        _write_hostfile(
            host_file, self.host, self.port, self.cert_path, self.authentication
        )

    def run(self):
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            ssl_keyfile=self.key_path,
            ssl_certfile=self.cert_path,
            ssl_version=ssl.PROTOCOL_SSLv23,
            ssl_keyfile_password=self.key_pw,
            log_level=logging.CRITICAL,
        )

    def _check_user(self, credentials: HTTPBasicCredentials) -> None:
        if credentials.password != self.authentication:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )

    def _log(self, request: Request) -> None:
        logging.getLogger("everserver").info(
            f"{request.scope['path']} entered from {request.client.host if request.client else 'unknown host'} with HTTP {request.method}"
        )

    def get_status(
        self, request: Request, credentials: HTTPBasicCredentials = Depends(security)
    ) -> JSONResponse:
        self._log(request)
        self._check_user(credentials)

        if self.state[STOP_ENDPOINT] == True:
            return JSONResponse(
                jsonable_encoder(
                    ExperimentRunnerStatus(status="Everest server stopped")
                )
            )

        if not self.runner:
            return JSONResponse(
                jsonable_encoder(
                    ExperimentRunnerStatus(status="Everest server is running")
                )
            )

        return JSONResponse(jsonable_encoder(self.runner.get_status()))

    def stop(
        self, request: Request, credentials: HTTPBasicCredentials = Depends(security)
    ) -> Response:
        self._log(request)
        self._check_user(credentials)
        self.state[STOP_ENDPOINT] = True
        return Response("Raise STOP flag succeeded. Everest initiates shutdown..", 200)

    def get_sim_progress(
        self, request: Request, credentials: HTTPBasicCredentials = Depends(security)
    ) -> JSONResponse:
        self._log(request)
        self._check_user(credentials)
        progress = self.state[SIM_PROGRESS_ENDPOINT]
        return JSONResponse(jsonable_encoder(progress))

    def get_opt_progress(
        self, request: Request, credentials: HTTPBasicCredentials = Depends(security)
    ) -> JSONResponse:
        self._log(request)
        self._check_user(credentials)
        progress = get_opt_status(self.optimization_output_dir)
        return JSONResponse(jsonable_encoder(progress))

    def start_experiment(
        self,
        config: EverestConfig,
        request: Request,
        credentials: HTTPBasicCredentials = Depends(security),
    ) -> Response:
        self._log(request)
        self._check_user(credentials)

        self.runner = ExperimentRunner(config, self.state)
        self.runner.start()

        return Response("Everest experiment started", 200)

    def get_state(
        self, request: Request, credentials: HTTPBasicCredentials = Depends(security)
    ) -> JSONResponse:
        self._log(request)
        self._check_user(credentials)
        return JSONResponse(jsonable_encoder(self.state))
