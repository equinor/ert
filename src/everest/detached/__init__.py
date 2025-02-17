import asyncio
import importlib
import json
import logging
import os
import re
import time
import traceback
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import polars as pl
import requests

from ert.scheduler import create_driver
from ert.scheduler.driver import Driver, FailedSubmit
from ert.scheduler.event import StartedEvent
from everest.config import EverestConfig, ServerConfig
from everest.everest_storage import EverestStorage
from everest.strings import (
    EVEREST_SERVER_CONFIG,
    OPT_PROGRESS_ENDPOINT,
    OPT_PROGRESS_ID,
    SIM_PROGRESS_ENDPOINT,
    SIM_PROGRESS_ID,
    START_EXPERIMENT_ENDPOINT,
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


async def start_server(
    config: EverestConfig, logging_level: int = logging.INFO
) -> Driver:
    """
    Start an Everest server running the optimization defined in the config
    """
    driver = create_driver(config.server.queue_system)  # type: ignore
    try:
        args = [
            "--output-dir",
            str(config.output_dir),
            "--logging-level",
            str(logging_level),
        ]
        poll_task = asyncio.create_task(driver.poll(), name="poll_task")
        await driver.submit(0, "everserver", *args, name=Path(config.config_file).stem)
    except FailedSubmit as err:
        raise ValueError(f"Failed to submit Everserver with error: {err}") from err
    status = await driver.event_queue.get()
    if not isinstance(status, StartedEvent):
        poll_task.cancel()
        raise ValueError(f"Everserver not started as expected, got status: {status}")
    poll_task.cancel()
    return driver


def stop_server(
    server_context: tuple[str, str, tuple[str, str]], retries: int = 5
) -> bool:
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


def start_experiment(
    server_context: tuple[str, str, tuple[str, str]],
    config: EverestConfig,
    retries: int = 5,
) -> None:
    for retry in range(retries):
        try:
            url, cert, auth = server_context
            start_endpoint = "/".join([url, START_EXPERIMENT_ENDPOINT])
            response = requests.post(
                start_endpoint,
                verify=cert,
                auth=auth,
                proxies=PROXY,  # type: ignore
                json=config.to_dict(),
            )
            response.raise_for_status()
            return
        except:
            logging.debug(traceback.format_exc())
            time.sleep(retry)
    raise RuntimeError("Failed to start experiment")


def extract_errors_from_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return re.findall(r"(Error \w+.*)", content)


def wait_for_server(output_dir: str, timeout: int | float) -> None:
    """
    Checks everest server has started _HTTP_REQUEST_RETRY times. Waits
    progressively longer between each check.

    Raise an exception when the timeout is reached.
    """
    sleep_time_increment = float(timeout) / (2**_HTTP_REQUEST_RETRY - 1)
    for retry_count in range(_HTTP_REQUEST_RETRY):
        if server_is_running(*ServerConfig.get_server_context(output_dir)):
            return
        else:
            time.sleep(sleep_time_increment * (2**retry_count))
    raise RuntimeError("Failed to get reply from server within configured timeout.")


def get_opt_status(output_folder: str) -> dict[str, Any]:
    """Return a dictionary with optimization information retrieved from storage"""
    if not Path(output_folder).exists() or not os.listdir(output_folder):
        return {}

    storage = EverestStorage(Path(output_folder))
    try:
        storage.read_from_output_dir()
    except FileNotFoundError:
        # Optimization output dir exists and not empty, but still missing
        # actual stored results
        return {}
    assert storage.data.objective_functions is not None
    assert storage.data.controls is not None
    objective_names = storage.data.objective_functions["objective_name"].to_list()
    control_names = storage.data.controls["control_name"].to_list()

    expected_objectives = pl.concat(
        [
            b.batch_objectives.select(objective_names)
            for b in storage.data.batches
            if b.batch_objectives is not None
        ]
    ).to_dict(as_series=False)

    expected_total_objective = [
        b.batch_objectives["total_objective_value"].item()
        for b in storage.data.batches
        if b.batch_objectives is not None
    ]

    improvement_batches = [b.batch_id for b in storage.data.batches if b.is_improvement]

    cli_monitor_data = {
        "batches": [
            b.batch_id
            for b in storage.data.batches
            if b.realization_controls is not None and b.batch_objectives is not None
        ],
        "controls": [
            b.realization_controls.select(control_names).to_dicts()[0]
            for b in storage.data.batches
            if b.realization_controls is not None
        ],
        "objective_value": expected_total_objective,
        "expected_objectives": expected_objectives,
    }

    return {
        "objective_history": expected_total_objective,
        "control_history": pl.concat(
            [
                b.realization_controls.select(control_names)
                for b in storage.data.batches
                if b.realization_controls is not None
            ]
        ).to_dict(as_series=False),
        "objectives_history": expected_objectives,
        "accepted_control_indices": improvement_batches,
        "cli_monitor_data": cli_monitor_data,
    }


def wait_for_server_to_stop(
    server_context: tuple[str, str, tuple[str, str]], timeout: int
) -> None:
    """
    Checks everest server has stopped _HTTP_REQUEST_RETRY times. Waits
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


def server_is_running(url: str, cert: str, auth: tuple[str, str]) -> bool:
    try:
        logging.info(f"Checking server status at {url} ")
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
    server_context: tuple[str, str, tuple[str, str]],
    callback: Callable[..., dict[str, Any]],
    polling_interval: int = 5,
) -> None:
    """
    Checks status on Everest server and calls callback when status changes

    Monitoring stops when the server stops answering. It can also be
    interrupted by returning True from the callback
    """
    url, cert, auth = server_context
    sim_endpoint = "/".join([url, SIM_PROGRESS_ENDPOINT])
    opt_endpoint = "/".join([url, OPT_PROGRESS_ENDPOINT])

    sim_status: dict[str, Any] = {}
    opt_status: dict[str, Any] = {}
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
    Path(importlib.util.find_spec("everest.detached").origin).parent  # type: ignore
    / os.path.join("jobs", EVEREST_SERVER_CONFIG)
)


_QUEUE_SYSTEMS: Mapping[Literal["LSF", "SLURM", "TORQUE"], dict[str, Any]] = {
    "LSF": {
        "options": [("options", "LSF_RESOURCE")],
        "name": "LSF_QUEUE",
    },
    "SLURM": {
        "options": [
            ("exclude_host", "EXCLUDE_HOST"),
            ("include_host", "INCLUDE_HOST"),
        ],
        "name": "PARTITION",
    },
    "TORQUE": {"options": ["cluster_label", "CLUSTER_LABEL"], "name": "QUEUE"},
}


def _query_server(cert: str, auth: tuple[str, str], endpoint: str) -> dict[str, Any]:
    """Retrieve data from an endpoint as a dictionary"""
    response = requests.get(endpoint, verify=cert, auth=auth, proxies=PROXY)  # type: ignore
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

    def default(self, o: ServerStatus | None) -> dict[str, str]:
        if type(o) is ServerStatus:
            return {"__enum__": str(o)}
        return json.JSONEncoder.default(self, o)

    @staticmethod
    def decode(obj: dict[str, str]) -> dict[str, str]:
        if "__enum__" in obj:
            _, member = obj["__enum__"].split(".")
            return getattr(ServerStatus, member)
        else:
            return obj


def update_everserver_status(
    everserver_status_path: str, status: ServerStatus, message: str | None = None
) -> None:
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


def everserver_status(everserver_status_path: str) -> dict[str, Any]:
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
