from __future__ import annotations

import logging
import queue
import ssl
import time
import traceback
from base64 import b64encode
from http import HTTPStatus
from pathlib import Path

import requests
from pydantic import ValidationError
from requests import HTTPError
from websockets.sync.client import connect

from _ert.threading import ErtThread
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models import BaseRunModelAPI
from ert.run_models.event import StatusEvents, status_event_from_json
from everest.strings import (
    CONFIG_PATH_ENDPOINT,
    SIMULATION_DIR_ENDPOINT,
    START_EXPERIMENT_ENDPOINT,
    START_TIME_ENDPOINT,
    STOP_ENDPOINT,
)

logger = logging.getLogger(__name__)


class EverestClient:
    def __init__(
        self,
        url: str,
        cert_file: str,
        username: str,
        password: str,
        ssl_context: ssl.SSLContext,
    ):
        self._url = url
        self._cert = cert_file
        self._username = username
        self._password = password
        self._ssl_context = ssl_context

        self._stop_endpoint = "/".join([url, STOP_ENDPOINT])
        self._start_endpoint = "/".join([url, START_EXPERIMENT_ENDPOINT])
        self._config_path_endpoint = "/".join([url, CONFIG_PATH_ENDPOINT])
        self._simulation_dir_endpoint = "/".join([url, SIMULATION_DIR_ENDPOINT])
        self._start_time_endpoint = "/".join([url, START_TIME_ENDPOINT])

        self._is_alive = False
        self._start_time: int | None = None

    @property
    def simulation_dir(self) -> str:
        return requests.get(
            self._simulation_dir_endpoint,
            verify=self._cert,
            auth=(self._username, self._password),
            proxies={"http": None, "https": None},  # type: ignore
        ).text

    @property
    def config_filename(self) -> str:
        config_path = requests.get(
            self._config_path_endpoint,
            verify=self._cert,
            auth=(self._username, self._password),
            proxies={"http": None, "https": None},  # type: ignore
        ).text

        return Path(config_path).name

    def get_runtime(self) -> int:
        if self._start_time is None:
            response = requests.get(
                self._start_time_endpoint,
                verify=self._cert,
                auth=(self._username, self._password),
                proxies={"http": None, "https": None},  # type: ignore
            )
            start_time = int(response.text)
            self._start_time = start_time

        return int(time.time()) - self._start_time

    @property
    def credentials(self) -> str:
        return b64encode(f"{self._username}:{self._password}".encode()).decode()

    def setup_event_queue_from_ws_endpoint(
        self,
        refresh_interval: float = 0.01,
        open_timeout: float = 30,
        websocket_recv_timeout: float = 1.0,
    ) -> tuple[queue.SimpleQueue[StatusEvents], ErtThread]:
        event_queue: queue.SimpleQueue[StatusEvents] = queue.SimpleQueue()

        def passthrough_ws_events() -> None:
            try:
                with connect(
                    self._url.replace("https://", "wss://") + "/events",
                    ssl=self._ssl_context,
                    open_timeout=open_timeout,
                    additional_headers={"Authorization": f"Basic {self.credentials}"},
                ) as websocket:
                    while not self._is_alive:
                        try:
                            message = websocket.recv(timeout=websocket_recv_timeout)
                        except TimeoutError:
                            message = None
                        if message:
                            try:
                                event = status_event_from_json(message)
                                event_queue.put(event)
                            except ValidationError as e:
                                logger.error(
                                    "Error when processing event %s", exc_info=e
                                )

                        time.sleep(refresh_interval)
            except Exception:
                logger.debug(traceback.format_exc())

        monitor_thread = ErtThread(
            name="everest_gui_event_monitor",
            target=passthrough_ws_events,
            daemon=True,
        )

        return event_queue, monitor_thread

    def create_run_model_api(self) -> BaseRunModelAPI:
        def start_fn(
            evaluator_server_config: EvaluatorServerConfig, restart: bool = False
        ) -> None:
            pass

        return BaseRunModelAPI(
            experiment_name=self.config_filename,
            runpath_format_string=self.simulation_dir,
            support_restart=False,
            start_simulations_thread=start_fn,
            cancel=self.stop,
            get_runtime=self.get_runtime,
            has_failed_realizations=lambda: False,
        )

    def stop(self) -> None:
        try:
            response = requests.post(
                self._stop_endpoint,
                verify=self._cert,
                auth=(self._username, self._password),
                proxies={"http": None, "https": None},  # type: ignore
            )

            if response.status_code == 200:
                logger.info("Cancelled experiment from Everest")
                print("Successfully cancelled experiment")
            else:
                logger.error(
                    f"Failed to cancel Everest experiment: POST @ {self._stop_endpoint}, "
                    f"server responded with status {response.status_code}: "
                    f"{HTTPStatus(response.status_code).phrase}"
                )
                print("Failed to cancel experiment")

        except requests.exceptions.ConnectionError as e:
            logger.error(
                f"Connection error when cancelling Everest experiment: {''.join(traceback.format_exception(e))}"
            )
            print("Failed to cancel experiment")

        except HTTPError as e:
            logger.error(
                f"HTTP error when cancelling Everest experiment: {''.join(traceback.format_exception(e))}"
            )
            print("Failed to cancel experiment")
