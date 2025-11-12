import asyncio
import logging
import re
import ssl
import time
import traceback
from base64 import b64encode
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests
from pydantic import ValidationError
from websockets import ConnectionClosedError, ConnectionClosedOK
from websockets.sync.client import connect

from ert.dark_storage.client import Client
from ert.run_models.event import EverestBatchResultEvent, status_event_from_json
from ert.scheduler import create_driver
from ert.scheduler.driver import Driver, FailedSubmit
from ert.scheduler.event import StartedEvent
from ert.trace import get_traceparent
from everest.config import EverestConfig, ServerConfig
from everest.strings import (
    OPT_PROGRESS_ID,
    SIM_PROGRESS_ID,
    EverEndpoints,
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
logger = logging.getLogger(__name__)


async def start_server(config: EverestConfig, logging_level: int) -> Driver:
    """
    Start an Everest server running the optimization defined in the config
    """
    driver = create_driver(config.server.queue_system, poll_period=0.1)  # type: ignore
    try:
        args = [
            "--output-dir",
            str(config.output_dir),
            "--logging-level",
            str(logging_level),
            "--traceparent",
            str(get_traceparent()),
        ]
        poll_task = asyncio.create_task(driver.poll(), name="poll_task")
        await driver.submit(
            0, "everserver", *args, name=f"{Path(config.config_file).stem}-server"
        )
    except FailedSubmit as err:
        raise ValueError(f"Failed to submit Everserver with error: {err}") from err
    status = await driver.event_queue.get()
    if not isinstance(status, StartedEvent):
        poll_task.cancel()
        raise ValueError(f"Everserver not started as expected, got status: {status}")
    poll_task.cancel()
    logger.debug(
        f"Everserver started. Events left in driver queue: {driver.event_queue.qsize()}"
    )
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
            stop_endpoint = "/".join([url, EverEndpoints.stop])
            response = requests.post(
                stop_endpoint,
                verify=cert,
                auth=auth,
                proxies=PROXY,  # type: ignore
            )
            response.raise_for_status()
        except Exception:
            logger.debug(traceback.format_exc())
            time.sleep(retry)
        else:
            return True
    return False


def start_experiment(
    server_context: tuple[str, str, tuple[str, str]],
    config: EverestConfig,
    retries: int = 5,
) -> None:
    for retry in range(retries):
        try:
            url, cert, auth = server_context
            start_endpoint = "/".join([url, EverEndpoints.start_experiment])
            response = requests.post(
                start_endpoint,
                verify=cert,
                auth=auth,
                proxies=PROXY,  # type: ignore
                json=config.to_dict(),
            )
            response.raise_for_status()
        except Exception:
            logger.debug(traceback.format_exc())
            time.sleep(retry)
        else:
            return
    raise RuntimeError("Failed to start experiment")


def extract_errors_from_file(path: str) -> list[str]:
    return re.findall(r"(Error \w+.*)", Path(path).read_text(encoding="utf-8"))


def wait_for_server(client: Client, timeout: int | float) -> None:
    """
    Waits until the everest server has started. Polls every second
    for server availability until timeout (measured in seconds).

    Raise an exception if no response within the timeout.
    """
    sleep_time: float = 1.0
    slept_time: float = 0.0
    while slept_time <= timeout:
        if server_is_running(
            *ServerConfig.get_server_context_from_conn_info(client.conn_info)
        ):
            logger.info(f"Waited {slept_time:g} seconds before everest server was up")
            return
        if slept_time + sleep_time > timeout:
            break
        time.sleep(sleep_time)
        slept_time += sleep_time
    raise RuntimeError(f"Failed to get reply from server within {slept_time:g} seconds")


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
        logger.debug(f"Checking server status at {url} ")
        if "None:None" in url:
            return False
        response = requests.get(
            url,
            verify=cert,
            auth=auth,
            timeout=1,
            proxies=PROXY,  # type: ignore
        )
        response.raise_for_status()
    except Exception:
        logger.debug(traceback.format_exc())
        return False
    return True


def get_opt_status_from_batch_result_event(
    event: EverestBatchResultEvent,
) -> dict[str, Any]:
    if not event.results:
        return {}

    assert event.batch is not None

    return {
        "batch": event.batch,
        "controls": event.results["controls"],
        "objective_value": event.results["total_objective_value"],
        "expected_objectives": event.results["objectives"],
    }


def start_monitor(
    server_context: tuple[str, str, tuple[str, str]],
    callback: Callable[..., None],
    polling_interval: float = 0.1,
) -> None:
    """
    Checks status on Everest server and calls callback when status changes

    Monitoring stops when the server stops answering.
    """
    url, cert, auth = server_context
    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(cafile=cert)
    username, password = auth
    credentials = b64encode(f"{username}:{password}".encode()).decode()

    try:
        with connect(
            url.replace("https://", "wss://") + "/events",
            ssl=ssl_context,
            open_timeout=30,
            additional_headers={"Authorization": f"Basic {credentials}"},
        ) as websocket:
            while True:
                try:
                    message = websocket.recv(timeout=1.0)
                    event = status_event_from_json(message)
                    if isinstance(event, EverestBatchResultEvent):
                        if event.result_type == "FunctionResult":
                            callback(
                                {
                                    OPT_PROGRESS_ID: get_opt_status_from_batch_result_event(  # noqa: E501
                                        event
                                    )
                                }
                            )
                    else:
                        callback({SIM_PROGRESS_ID: event})
                except TimeoutError:
                    pass
                except ConnectionClosedOK:
                    logger.debug("Connection closed")
                    break
                except ConnectionClosedError:
                    logger.debug("Connection closed")
                    break
                except ValidationError as e:
                    logger.error("Error when processing event %s", exc_info=e)

                time.sleep(polling_interval)

    except Exception:
        logger.exception(traceback.format_exc())
