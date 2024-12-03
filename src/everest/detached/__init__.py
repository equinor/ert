import asyncio
import importlib
import json
import logging
import os
import re
import time
import traceback
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Literal

import requests
from seba_sqlite.exceptions import ObjectNotFoundError
from seba_sqlite.snapshot import SebaSnapshot

from ert.config import QueueSystem
from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    QueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
    activate_script,
)
from ert.plugins import ErtPluginManager
from ert.scheduler import create_driver
from ert.scheduler.driver import Driver, FailedSubmit
from ert.scheduler.event import StartedEvent
from everest.config import EverestConfig, ServerConfig, SimulatorConfig
from everest.config_keys import ConfigKeys as CK
from everest.strings import (
    EVEREST_SERVER_CONFIG,
    OPT_PROGRESS_ENDPOINT,
    OPT_PROGRESS_ID,
    SIM_PROGRESS_ENDPOINT,
    SIM_PROGRESS_ID,
    STOP_ENDPOINT,
)

# Specifies how many times to try a http request within the specified timeout.
_HTTP_REQUEST_RETRY = 10

# Proxy configuration for outgoing requests.
# For internal LAN HTTP requests not using a proxy is recommended.
PROXY = {"http": None, "https": None}

# The methods in this file are typically called for the client side.
# Information from the client side is relatively uninteresting, so we show it in
# the default logger (stdout). Info from the server will be logged to the
# everest.log file instead


async def start_server(config: EverestConfig, debug: bool = False) -> Driver:
    """
    Start an Everest server running the optimization defined in the config
    """
    driver = create_driver(get_server_queue_options(config.simulator, config.server))
    try:
        args = ["--config-file", str(config.config_path)]
        if debug:
            args.append("--debug")
        poll_task = asyncio.create_task(driver.poll(), name="poll_task")
        await driver.submit(0, "everserver", *args)
    except FailedSubmit as err:
        raise ValueError(f"Failed to submit Everserver with error: {err}") from err
    status = await driver.event_queue.get()
    if not isinstance(status, StartedEvent):
        poll_task.cancel()
        raise ValueError(f"Everserver not started as expected, got status: {status}")
    poll_task.cancel()
    return driver


def stop_server(server_context: tuple[str, str, tuple[str, str]], retries: int = 5):
    """
    Stop server if found and it is running.
    """
    for retry in range(retries):
        try:
            url, cert, auth = server_context
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
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return re.findall(r"(Error \w+.*)", content)


def wait_for_server(output_dir: str, timeout: int) -> None:
    """
    Checks everest server has started _HTTP_REQUEST_RETRY times. Waits
    progressively longer between each check.

    Raise an exception when the timeout is reached.
    """
    everserver_status_path = ServerConfig.get_everserver_status_path(output_dir)
    if not server_is_running(*ServerConfig.get_server_context(output_dir)):
        sleep_time_increment = float(timeout) / (2**_HTTP_REQUEST_RETRY - 1)
        for retry_count in range(_HTTP_REQUEST_RETRY):
            # Failure may occur before contact with the server is established:
            status = everserver_status(everserver_status_path)
            if status["status"] == ServerStatus.completed:
                # For very small cases the optimization will finish and bring down the
                # server before we can verify that it is running.
                return

            if status["status"] == ServerStatus.failed:
                raise SystemExit(
                    "Failed to start Everest with error:\n{}".format(status["message"])
                )

            sleep_time = sleep_time_increment * (2**retry_count)
            time.sleep(sleep_time)
            if server_is_running(*ServerConfig.get_server_context(output_dir)):
                return

    # If number of retries reached and server is not running - throw exception
    raise RuntimeError("Failed to start server within configured timeout.")


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


def wait_for_server_to_stop(server_context: tuple[str, str, tuple[str, str]], timeout):
    """
    Checks everest server has stoped _HTTP_REQUEST_RETRY times. Waits
    progressively longer between each check.

    Raise an exception when the timeout is reached.
    """
    if server_is_running(*server_context):
        sleep_time_increment = float(timeout) / (2**_HTTP_REQUEST_RETRY - 1)
        for retry_count in range(_HTTP_REQUEST_RETRY):
            sleep_time = sleep_time_increment * (2**retry_count)
            time.sleep(sleep_time)
            if not server_is_running(*server_context):
                return

    # If number of retries reached and server still running - throw exception
    if server_is_running(*server_context):
        raise Exception("Failed to stop server within configured timeout.")


def server_is_running(url: str, cert: str, auth: tuple[str, str]):
    try:
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


def start_monitor(
    server_context: tuple[str, str, tuple[str, str]], callback, polling_interval=5
):
    """
    Checks status on Everest server and calls callback when status changes

    Monitoring stops when the server stops answering. It can also be
    interrupted by returning True from the callback
    """
    url, cert, auth = server_context
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
    "TORQUE": {"options": [CK.TORQUE_CLUSTER_LABEL, "CLUSTER_LABEL"], "name": "QUEUE"},
}


def _find_res_queue_system(
    simulator: SimulatorConfig | None,
    server: ServerConfig | None,
):
    queue_system_simulator: Literal["lsf", "local", "slurm", "torque"] = "local"
    if simulator is not None and simulator.queue_system is not None:
        queue_system_simulator = simulator.queue_system

    queue_system = queue_system_simulator
    if server is not None:
        queue_system = server.queue_system or queue_system

    if queue_system_simulator == CK.LOCAL and queue_system_simulator != queue_system:
        raise ValueError(
            f"The simulator is using {CK.LOCAL} as queue system "
            f"while the everest server is using {queue_system}. "
            f"If the simulator is using {CK.LOCAL}, so must the everest server."
        )

    assert queue_system is not None
    return QueueSystem(queue_system.upper())


def get_server_queue_options(
    simulator: SimulatorConfig | None,
    server: ServerConfig | None,
) -> QueueOptions:
    script = ErtPluginManager().activate_script() or activate_script()
    queue_system = _find_res_queue_system(simulator, server)
    ever_queue_config = server if server is not None else simulator
    if queue_system == QueueSystem.LSF:
        queue = LsfQueueOptions(
            activate_script=script,
            lsf_queue=ever_queue_config.name,
            lsf_resource=ever_queue_config.options,
        )
    elif queue_system == QueueSystem.SLURM:
        queue = SlurmQueueOptions(
            activate_script=script,
            exclude_host=ever_queue_config.exclude_host,
            include_host=ever_queue_config.include_host,
            partition=ever_queue_config.name,
        )
    elif queue_system == QueueSystem.TORQUE:
        queue = TorqueQueueOptions(
            activate_script=script,
            queue=ever_queue_config.name,
            keep_qsub_output=ever_queue_config.keep_qsub_output,
        )
    elif queue_system == QueueSystem.LOCAL:
        queue = LocalQueueOptions()
    else:
        raise ValueError(f"Unknown queue system: {queue_system}")
    queue.max_running = 1
    return queue


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
    everserver_status_path: str, status: ServerStatus, message: str | None = None
):
    """Update the everest server status with new status information"""
    new_status = {"status": status, "message": message}
    path = everserver_status_path
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as outfile:
            json.dump(new_status, outfile, cls=ServerStatusEncoder)
    elif os.path.exists(path):
        server_status = everserver_status(path)
        if server_status["message"] is not None:
            if message is not None:
                new_status["message"] = "{}\n{}".format(
                    server_status["message"], message
                )
            else:
                new_status["message"] = server_status["message"]
        with open(path, "w", encoding="utf-8") as outfile:
            json.dump(new_status, outfile, cls=ServerStatusEncoder)


def everserver_status(everserver_status_path: str):
    """Returns a dictionary representing the everest server status. If the
    status file is not found we assume the server has never ran before, and will
    return a status of ServerStatus.never_run

    Example: {
                'status': ServerStatus.completed
                'message': None
             }
    """
    if os.path.exists(everserver_status_path):
        with open(everserver_status_path, encoding="utf-8") as f:
            return json.load(f, object_hook=ServerStatusEncoder.decode)
    else:
        return {"status": ServerStatus.never_run, "message": None}
