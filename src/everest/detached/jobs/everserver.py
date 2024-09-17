import argparse
import json
import logging
import os
import socket
import ssl
import threading
import traceback
from base64 import b64encode
from datetime import datetime, timedelta
from functools import partial, wraps

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from dns import resolver, reversename
from flask import Flask, Response, jsonify, request
from ropt.enums import OptimizerExitCode

from everest import export_to_csv, start_optimization, validate_export
from everest.config import EverestConfig
from everest.detached import ServerStatus, get_opt_status, update_everserver_status
from everest.simulator import JOB_FAILURE
from everest.strings import (
    EVEREST,
    OPT_FAILURE_REALIZATIONS,
    OPT_PROGRESS_ENDPOINT,
    SIM_PROGRESS_ENDPOINT,
    STOP_ENDPOINT,
)
from everest.util import configure_logger, makedirs_if_needed, version_info


def get_machine_name():
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
        logging.debug(traceback.format_exc())
        return "localhost"


def _sim_monitor(context_status, event=None, shared_data=None):
    status = context_status["status"]
    shared_data[SIM_PROGRESS_ENDPOINT] = {
        "batch_number": context_status["batch_number"],
        "status": {
            "running": status.running,
            "waiting": status.waiting,
            "pending": status.pending,
            "complete": status.complete,
            "failed": status.failed,
        },
        "progress": context_status["progress"],
        "event": event,
    }

    if shared_data[STOP_ENDPOINT]:
        return "stop_queue"


def _opt_monitor(shared_data=None):
    if shared_data[STOP_ENDPOINT]:
        return "stop_optimization"


def _everserver_thread(shared_data, server_config):
    app = Flask(__name__)

    def check_user(password):
        return password == server_config["authentication"]

    def requires_authenticated(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            auth = request.authorization
            if not auth or not check_user(auth.password):
                return "unauthorized", 401
            return f(*args, **kwargs)

        return decorated

    def log(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            url = request.path
            method = request.method
            ip = request.environ.get("HTTP_X_REAL_IP", request.remote_addr)
            logging.getLogger("everserver").info(
                "{} entered from {} with HTTP {}".format(url, ip, method)
            )
            return f(*args, **kwargs)

        return decorated

    @app.route("/")
    @requires_authenticated
    @log
    def get_home():
        return "Everest is running"

    @app.route("/" + STOP_ENDPOINT, methods=["POST"])
    @requires_authenticated
    @log
    def stop():
        shared_data[STOP_ENDPOINT] = True
        return Response("Raise STOP flag succeeded. Everest initiates shutdown..", 200)

    @app.route("/" + SIM_PROGRESS_ENDPOINT)
    @requires_authenticated
    @log
    def get_sim_progress():
        return jsonify(shared_data[SIM_PROGRESS_ENDPOINT])

    @app.route("/" + OPT_PROGRESS_ENDPOINT)
    @requires_authenticated
    @log
    def get_opt_progress():
        progress = get_opt_status(server_config["optimization_output_dir"])
        return jsonify(progress)

    ctx = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
    ctx.load_cert_chain(
        server_config["cert_path"],
        server_config["key_path"],
        server_config["key_passwd"],
    )
    app.run(host="0.0.0.0", port=server_config["port"], ssl_context=ctx)


def _find_open_port(host, lower, upper):
    for port in range(lower, upper):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, port))
            sock.close()
            return port
        except socket.error:
            logging.getLogger("everserver").info(
                "Port {} for host {} is taken".format(port, host)
            )
    msg = "No open port for host {} in the range {}-{}".format(host, lower, upper)
    logging.getLogger("everserver").exception(msg)
    raise Exception(msg)


def _write_hostfile(config: EverestConfig, host, port, cert, auth):
    host_file_path = config.hostfile_path
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


def _configure_loggers(config: EverestConfig):
    detached_node_dir = config.detached_node_dir
    everest_logs_dir = config.log_dir

    configure_logger(
        name="res",
        file_path=os.path.join(detached_node_dir, "simulations.log"),
        log_level=logging.INFO,
    )

    configure_logger(
        name="everserver",
        file_path=os.path.join(detached_node_dir, "endpoint.log"),
        log_level=logging.INFO,
    )

    configure_logger(
        name=EVEREST,
        file_path=os.path.join(everest_logs_dir, "everest.log"),
        log_level=config.logging_level,
        log_to_azure=True,
    )

    configure_logger(
        name="forward_models",
        file_path=os.path.join(everest_logs_dir, "forward_models.log"),
        log_level=config.logging_level,
    )

    configure_logger(
        name="ropt",
        file_path=os.path.join(everest_logs_dir, "ropt.log"),
        log_level=config.logging_level,
    )


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config-file", type=str)
    arg_parser.add_argument("--debug", action="store_true")
    options = arg_parser.parse_args()
    config = EverestConfig.load_file(options.config_file)
    if options.debug:
        config.logging_level = "debug"

    try:
        _configure_loggers(config)
        update_everserver_status(config, ServerStatus.starting)
        logging.getLogger(EVEREST).info(version_info())
        logging.getLogger(EVEREST).info(
            "Output directory: {}".format(config.output_dir)
        )
        logging.getLogger(EVEREST).debug(str(options))

        authentication = _generate_authentication()
        cert_path, key_path, key_pw = _generate_certificate(config)
        host = get_machine_name()
        port = _find_open_port(host, lower=5000, upper=5800)
        _write_hostfile(config, host, port, cert_path, authentication)

        shared_data = {
            SIM_PROGRESS_ENDPOINT: {},
            STOP_ENDPOINT: False,
        }

        server_config = {
            "optimization_output_dir": config.optimization_output_dir,
            "port": port,
            "cert_path": cert_path,
            "key_path": key_path,
            "key_passwd": key_pw,
            "authentication": authentication,
        }

        everserver_instance = threading.Thread(
            target=_everserver_thread,
            args=(shared_data, server_config),
        )
        everserver_instance.daemon = True
        everserver_instance.start()
    except:
        update_everserver_status(
            config, ServerStatus.failed, message=traceback.format_exc()
        )
        return

    try:
        update_everserver_status(config, ServerStatus.running)
        exit_code = start_optimization(
            config,
            simulation_callback=partial(_sim_monitor, shared_data=shared_data),
            optimization_callback=partial(_opt_monitor, shared_data=shared_data),
        )
        status, message = _get_optimization_status(exit_code, shared_data)
        if status != ServerStatus.completed:
            update_everserver_status(config, status, message)
            return
    except:
        if shared_data[STOP_ENDPOINT]:
            update_everserver_status(
                config, ServerStatus.stopped, message="Optimization aborted."
            )
        else:
            update_everserver_status(
                config, ServerStatus.failed, message=traceback.format_exc()
            )
        return

    try:
        # Exporting data
        update_everserver_status(config, ServerStatus.exporting_to_csv)
        err_msgs, export_ecl = validate_export(config)
        for msg in err_msgs:
            logging.getLogger(EVEREST).warning(msg)
        export_to_csv(config, export_ecl=export_ecl)
    except:
        update_everserver_status(
            config, ServerStatus.failed, message=traceback.format_exc()
        )
        return

    update_everserver_status(config, ServerStatus.completed, message=message)


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
            (
                job["name"]
                for queue in shared_data[SIM_PROGRESS_ENDPOINT]["progress"]
                for job in queue
                if job["status"] == JOB_FAILURE
            )
        ).keys()
        messages.append(
            "{} job failures caused by: {}".format(failed, ", ".join(failed_jobs))
        )
    return messages


def _generate_certificate(config: EverestConfig):
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
    cert_name = get_machine_name()
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
    cert_folder = config.certificate_dir
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


def _generate_authentication():
    n_bytes = 128
    random_bytes = bytes(os.urandom(n_bytes))
    return b64encode(random_bytes).decode("utf-8")
