import importlib
import json
import logging
import os
import re
import time
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Literal, Mapping, Optional, Tuple

import requests
from seba_sqlite.exceptions import ObjectNotFoundError
from seba_sqlite.snapshot import SebaSnapshot

from ert import BatchContext, BatchSimulator
from ert.config import ErtConfig, QueueSystem
from everest.config import EverestConfig
from everest.config_keys import ConfigKeys as CK
from everest.simulator import JOB_FAILURE, JOB_SUCCESS, Status
from everest.strings import (
    EVEREST,
    EVEREST_SERVER_CONFIG,
    OPT_PROGRESS_ENDPOINT,
    OPT_PROGRESS_ID,
    SIM_PROGRESS_ENDPOINT,
    SIM_PROGRESS_ID,
    SIMULATION_DIR,
    STOP_ENDPOINT,
)
from everest.util import configure_logger

# Specifies how many times to try a http request within the specified timeout.
_HTTP_REQUEST_RETRY = 10

# Proxy configuration for outgoing requests.
# For internal LAN HTTP requests not using a proxy is recommended.
PROXY = {"http": None, "https": None}

# The methods in this file are typically called for the client side.
# Information from the client side is relatively uninteresting, so we show it in
# the default logger (stdout). Info from the server will be logged to the
# everest.log file instead


# The Everest server is launched through ert. When running on LSF everything
# works fine. But when running on the local queue (mainly for testing and
# debugging) the ert causes all sorts of problems if the server or the
# context go out of scope. So we keep them alive for now.
# Note that, after the server is stopped (eg by a call to stop_server), the
# context does not immediately terminate. The method _context_stop_and_wait
# stops the context (if available) and waits until the context is terminated
# (to be used typically in tests)
_server = None
_context = None


def start_server(config: EverestConfig, ert_config: ErtConfig, storage):
    """
    Start an Everest server running the optimization defined in the config
    """
    if server_is_running(config):  # better safe than sorry
        return

    log_dir = config.log_dir

    configure_logger(
        name="res",
        file_path=os.path.join(log_dir, "everest_server.log"),
        log_level=logging.INFO,
        log_to_azure=True,
    )

    configure_logger(
        name=__name__,
        file_path=os.path.join(log_dir, "simulations.log"),
        log_level=logging.INFO,
    )

    global _server  # noqa: PLW0603
    global _context  # noqa: PLW0603
    if _context and _context.running():
        raise RuntimeError(
            "Starting two instances of everest server "
            "in the same process is not allowed!"
        )

    try:
        _save_running_config(config)
    except (OSError, LookupError) as e:
        logging.getLogger(EVEREST).error(
            "Failed to save optimization config: {}".format(e)
        )

    experiment = storage.create_experiment(
        name=f"DetachedEverest@{datetime.now().strftime('%Y-%m-%d@%H:%M:%S')}",
        parameters=[],
        responses=[],
    )

    _server = BatchSimulator(ert_config, {}, [])
    _context = _server.start("dispatch_server", [(0, {})], experiment)

    return _context


def _save_running_config(config: EverestConfig):
    assert config.output_dir is not None
    assert config.config_file is not None
    save_config_path = os.path.join(config.output_dir, config.config_file)
    config.dump(save_config_path)


def context_stop_and_wait():
    global _context  # noqa: PLW0602
    if _context:
        _context.stop()
        while _context.running():
            time.sleep(1)


def wait_for_context():
    global _context  # noqa: PLW0602
    if _context and _context.running():
        while _context.running():
            time.sleep(1)


def stop_server(config: EverestConfig, retries: int = 5):
    """
    Stop server if found and it is running.
    """
    for retry in range(retries):
        try:
            url, cert, auth = config.server_context
            stop_endpoint = "/".join([url, STOP_ENDPOINT])
            response = requests.post(
                stop_endpoint,
                verify=cert,
                auth=auth,
                proxies=PROXY,  # type: ignore
            )
            response.raise_for_status()
            return True
        except:
            logging.debug(traceback.format_exc())
            time.sleep(retry)
    return False


def extract_errors_from_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return re.findall(r"(Error \w+.*)", content)


def wait_for_server(
    config: EverestConfig, timeout: int, context: Optional[BatchContext] = None
) -> None:
    """
    Checks everest server has started _HTTP_REQUEST_RETRY times. Waits
    progressively longer between each check.

    Raise an exception when the timeout is reached.
    """
    if not server_is_running(config):
        sleep_time_increment = float(timeout) / (2**_HTTP_REQUEST_RETRY - 1)
        for retry_count in range(_HTTP_REQUEST_RETRY):
            # Failure may occur before contact with the server is established:
            status = everserver_status(config)
            if status["status"] == ServerStatus.completed:
                # For very small cases the optimization will finish and bring down the
                # server before we can verify that it is running.
                return

            if status["status"] == ServerStatus.failed:
                raise SystemExit(
                    "Failed to start Everest with error:\n{}".format(status["message"])
                )
            # Job queueing may fail:
            if context is not None and context.has_job_failed(0):
                path = context.job_progress(0).steps[0].std_err_file
                for err in extract_errors_from_file(path):
                    update_everserver_status(config, ServerStatus.failed, message=err)
                    logging.error(err)
                raise SystemExit("Failed to start Everest server.")
            sleep_time = sleep_time_increment * (2**retry_count)
            time.sleep(sleep_time)
            if server_is_running(config):
                return

    # If number of retries reached and server is not running - throw exception
    if not server_is_running(config):
        raise Exception("Failed to start server within configured timeout.")


def get_sim_status(config: EverestConfig):
    """Retrieve a seba database snapshot and return a list of simulation
    information objects for each of the available batches in the database

    Example: [{progress: [[{'start_time': u'Thu, 16 May 2019 16:53:20  UTC',
                            'end_time': u'Thu, 16 May 2019 16:53:20  UTC',
                            'status': JOB_SUCCESS}]],
               'batch_number': 0,
               'event': 'update'}, ..]
    """

    seba_snapshot = SebaSnapshot(config.optimization_output_dir)
    snapshot = seba_snapshot.get_snapshot()

    def timestamp2str(timestamp):
        if timestamp:
            return "{} UTC".format(
                datetime.fromtimestamp(timestamp).strftime("%a, %d %b %Y %H:%M:%S %Z")
            )
        else:
            return None

    sim_progress: dict = {}
    for sim in snapshot.simulation_data:
        sim_metadata = {
            "start_time": timestamp2str(sim.start_time),
            "end_time": timestamp2str(sim.end_time),
            "realization": sim.realization,
            "simulation": sim.simulation,
            "status": JOB_SUCCESS if sim.success else JOB_FAILURE,
        }
        if sim.batch in sim_progress:
            sim_progress[sim.batch]["progress"].append([sim_metadata])
        else:
            sim_progress[sim.batch] = {
                "progress": [[sim_metadata]],
                "batch_number": sim.batch,
                "event": "update",
            }
    for status in sim_progress.values():
        fm_runs = len(status["progress"])
        failed = sum(
            fm_run[0]["status"] == JOB_FAILURE for fm_run in status["progress"]
        )
        status.update(
            {
                "status": Status(
                    waiting=0,
                    pending=0,
                    running=0,
                    failed=failed,
                    complete=fm_runs - failed,
                )
            }
        )

    return list(sim_progress.values())


def get_opt_status(output_folder):
    """Retrieve a seba database snapshot and return a dictionary with
    optimization information."""
    if not os.path.exists(os.path.join(output_folder, "seba.db")):
        return {}
    try:
        seba_snapshot = SebaSnapshot(output_folder)
    except ObjectNotFoundError:
        return {}
    snapshot = seba_snapshot.get_snapshot(filter_out_gradient=True)

    cli_monitor_data = {}
    if snapshot.optimization_data:
        cli_monitor_data = {
            "batches": [item.batch_id for item in snapshot.optimization_data],
            "controls": [item.controls for item in snapshot.optimization_data],
            "objective_value": [
                item.objective_value for item in snapshot.optimization_data
            ],
            "expected_objectives": snapshot.expected_objectives,
        }

    return {
        "objective_history": snapshot.expected_single_objective,
        "control_history": snapshot.optimization_controls,
        "objectives_history": snapshot.expected_objectives,
        "accepted_control_indices": snapshot.increased_merit_indices,
        "cli_monitor_data": cli_monitor_data,
    }


def wait_for_server_to_stop(config: EverestConfig, timeout):
    """
    Checks everest server has stoped _HTTP_REQUEST_RETRY times. Waits
    progressively longer between each check.

    Raise an exception when the timeout is reached.
    """
    if server_is_running(config):
        sleep_time_increment = float(timeout) / (2**_HTTP_REQUEST_RETRY - 1)
        for retry_count in range(_HTTP_REQUEST_RETRY):
            sleep_time = sleep_time_increment * (2**retry_count)
            time.sleep(sleep_time)
            if not server_is_running(config):
                return

    # If number of retries reached and server still running - throw exception
    if server_is_running(config):
        raise Exception("Failed to stop server within configured timeout.")


def server_is_running(config: EverestConfig):
    try:
        url, cert, auth = config.server_context
        response = requests.get(
            url,
            verify=cert,
            auth=auth,
            timeout=1,
            proxies=PROXY,  # type: ignore
        )
        response.raise_for_status()
    except:
        logging.debug(traceback.format_exc())
        return False
    return True


def get_optimization_status(config: EverestConfig):
    seba_snapshot = SebaSnapshot(config.optimization_output_dir)
    snapshot = seba_snapshot.get_snapshot(filter_out_gradient=True)

    return {
        "objective_history": snapshot.expected_single_objective,
        "control_history": snapshot.optimization_controls,
    }


def start_monitor(config: EverestConfig, callback, polling_interval=5):
    """
    Checks status on Everest server and calls callback when status changes

    Monitoring stops when the server stops answering. It can also be
    interrupted by returning True from the callback
    """
    url, cert, auth = config.server_context
    sim_endpoint = "/".join([url, SIM_PROGRESS_ENDPOINT])
    opt_endpoint = "/".join([url, OPT_PROGRESS_ENDPOINT])

    sim_status: dict = {}
    opt_status: dict = {}
    stop = False

    try:
        while not stop:
            new_sim_status = _query_server(cert, auth, sim_endpoint)
            if new_sim_status != sim_status:
                sim_status = new_sim_status
                ret = bool(callback({SIM_PROGRESS_ID: sim_status}))
                stop |= ret
            # When the API will support it query only from a certain batch on

            # Check the optimization status
            new_opt_status = _query_server(cert, auth, opt_endpoint)
            if new_opt_status != opt_status:
                opt_status = new_opt_status
                ret = bool(callback({OPT_PROGRESS_ID: opt_status}))
                stop |= ret
            time.sleep(polling_interval)
    except:
        logging.debug(traceback.format_exc())


_EVERSERVER_JOB_PATH = str(
    Path(importlib.util.find_spec("everest.detached").origin).parent
    / os.path.join("jobs", EVEREST_SERVER_CONFIG)
)


_QUEUE_SYSTEMS: Mapping[Literal["LSF", "SLURM"], dict] = {
    "LSF": {
        "options": [(CK.LSF_OPTIONS, "LSF_RESOURCE")],
        "name": "LSF_QUEUE",
    },
    "SLURM": {
        "options": [
            (CK.SLURM_EXCLUDE_HOST_OPTION, "EXCLUDE_HOST"),
            (CK.SLURM_INCLUDE_HOST_OPTION, "INCLUDE_HOST"),
        ],
        "name": "PARTITION",
    },
}


def _add_simulator_defaults(
    options,
    config: EverestConfig,
    queue_options: List[Tuple[str, str]],
    queue_system: Literal["LSF", "SLURM"],
):
    simulator_options = (
        config.simulator.extract_ert_queue_options(
            queue_system=queue_system, everest_to_ert_key_tuples=queue_options
        )
        if config.simulator is not None
        else []
    )

    option_names = [option[1] for option in options]
    simulator_option_names = [option[1] for option in simulator_options]
    options.extend(
        simulator_options[simulator_option_names.index(res_key)]
        for _, res_key in queue_options
        if res_key not in option_names and res_key in simulator_option_names
    )
    return options


def _generate_queue_options(
    config: EverestConfig,
    queue_options: List[Tuple[str, str]],
    res_queue_name: str,  # Literal["LSF_QUEUE", "PARTITION"]?
    queue_system: Literal["LSF", "SLURM"],
):
    queue_name_simulator = (
        config.simulator.name if config.simulator is not None else None
    )

    queue_name = config.server.name if config.server is not None else None

    if queue_name is None:
        queue_name = queue_name_simulator

    options = (
        config.server.extract_ert_queue_options(
            queue_system=queue_system, everest_to_ert_key_tuples=queue_options
        )
        if config.server is not None
        else [(queue_system, "MAX_RUNNING", 1)]
    )

    if queue_name:
        options.append(
            (
                queue_system,
                res_queue_name,
                queue_name,
            ),
        )
    # Inherit the include/exclude_host from the simulator config entry, if necessary.
    # Currently this is only used by the slurm driver.
    if queue_system == "SLURM":
        options = _add_simulator_defaults(options, config, queue_options, queue_system)
    return options


def _find_res_queue_system(config: EverestConfig):
    queue_system_simulator: Literal["lsf", "local", "slurm"] = "local"
    if config.simulator is not None:
        queue_system_simulator = config.simulator.queue_system or queue_system_simulator

    queue_system = queue_system_simulator
    if config.server is not None:
        queue_system = config.server.queue_system or queue_system

    if queue_system_simulator == CK.LOCAL and queue_system_simulator != queue_system:
        raise ValueError(
            f"The simulator is using {CK.LOCAL} as queue system "
            f"while the everest server is using {queue_system}. "
            f"If the simulator is using {CK.LOCAL}, so must the everest server."
        )

    assert queue_system is not None
    return QueueSystem(queue_system.upper())


def generate_everserver_ert_config(config: EverestConfig, debug_mode: bool = False):
    assert config.config_directory is not None
    assert config.config_file is not None

    site_config = ErtConfig.read_site_config()
    abs_everest_config = os.path.join(config.config_directory, config.config_file)
    detached_node_dir = config.detached_node_dir
    simulation_path = os.path.join(detached_node_dir, SIMULATION_DIR)
    queue_system = _find_res_queue_system(config)
    arg_list = ["--config-file", abs_everest_config]
    if debug_mode:
        arg_list.append("--debug")

    everserver_config = {} if site_config is None else site_config
    everserver_config.update(
        {
            "RUNPATH": simulation_path,
            "JOBNAME": EVEREST_SERVER_CONFIG,
            "NUM_REALIZATIONS": 1,
            "MAX_SUBMIT": 1,
            "ENSPATH": os.path.join(detached_node_dir, EVEREST_SERVER_CONFIG),
            "RUNPATH_FILE": os.path.join(detached_node_dir, ".res_runpath_list"),
        }
    )
    install_job = everserver_config.get("INSTALL_JOB", [])
    install_job.append((EVEREST_SERVER_CONFIG, _EVERSERVER_JOB_PATH))
    everserver_config["INSTALL_JOB"] = install_job

    simulation_job = everserver_config.get("SIMULATION_JOB", [])
    simulation_job.append([EVEREST_SERVER_CONFIG, *arg_list])
    everserver_config["SIMULATION_JOB"] = simulation_job

    if queue_system in _QUEUE_SYSTEMS:
        everserver_config["QUEUE_SYSTEM"] = queue_system
        queue_options = _generate_queue_options(
            config,
            _QUEUE_SYSTEMS[queue_system]["options"],
            _QUEUE_SYSTEMS[queue_system]["name"],
            queue_system,
        )
        if queue_options:
            everserver_config.setdefault("QUEUE_OPTION", []).extend(queue_options)
    else:
        everserver_config["QUEUE_SYSTEM"] = queue_system

    return everserver_config


def _query_server(cert, auth, endpoint):
    """Retrieve data from an endpoint as a dictionary"""
    response = requests.get(endpoint, verify=cert, auth=auth, proxies=PROXY)
    response.raise_for_status()
    return response.json()


class ServerStatus(Enum):
    """Keep track of the different states the everest server is in"""

    starting = 1
    running = 2
    exporting_to_csv = 3
    completed = 4
    stopped = 5
    failed = 6
    never_run = 7


class ServerStatusEncoder(json.JSONEncoder):
    """Facilitates encoding and decoding the server status enum object to
    and from a json file"""

    def default(self, o):
        if type(o) is ServerStatus:
            return {"__enum__": str(o)}
        return json.JSONEncoder.default(self, o)

    @staticmethod
    def decode(obj):
        if "__enum__" in obj:
            _, member = obj["__enum__"].split(".")
            return getattr(ServerStatus, member)
        else:
            return obj


def update_everserver_status(
    config: EverestConfig, status: ServerStatus, message: Optional[str] = None
):
    """Update the everest server status with new status information"""
    new_status = {"status": status, "message": message}
    path = config.everserver_status_path
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as outfile:
            json.dump(new_status, outfile, cls=ServerStatusEncoder)
    elif os.path.exists(path):
        server_status = everserver_status(config)
        if server_status["message"] is not None:
            if message is not None:
                new_status["message"] = "{}\n{}".format(
                    server_status["message"], message
                )
            else:
                new_status["message"] = server_status["message"]
        with open(path, "w", encoding="utf-8") as outfile:
            json.dump(new_status, outfile, cls=ServerStatusEncoder)


def everserver_status(config: EverestConfig):
    """Returns a dictionary representing the everest server status. If the
    status file is not found we assume the server has never ran before, and will
    return a status of ServerStatus.never_run

    Example: {
                'status': ServerStatus.completed
                'message': None
             }
    """
    path = config.everserver_status_path
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f, object_hook=ServerStatusEncoder.decode)
    else:
        return {"status": ServerStatus.never_run, "message": None}
