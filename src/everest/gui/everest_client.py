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
from websockets.exceptions import ConnectionClosedError
from websockets.sync.client import connect

from _ert.threading import ErtThread
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models import RunModelAPI
from ert.run_models.event import StatusEvents, status_event_from_json
from everest.strings import EverEndpoints

logger = logging.getLogger(__name__)


class EverestClient:
    def __init__(
        self,
        url: str,
        cert_file: str,
        username: str,
        password: str,
        ssl_context: ssl.SSLContext,
    ) -> None:
        self._url = url
        self._cert = cert_file
        self._username = username
        self._password = password
        self._ssl_context = ssl_context

        self._is_alive = False
        self._start_time: int | None = None

    def _http_get(self, endpoint: EverEndpoints) -> requests.Response:
        return requests.get(
            "/".join([self._url, endpoint]),
            verify=self._cert,
            auth=(self._username, self._password),
            proxies={"http": None, "https": None},  # type: ignore
        )

    def _http_post(self, endpoint: EverEndpoints) -> requests.Response:
        return requests.post(
            "/".join([self._url, endpoint]),
            verify=self._cert,
            auth=(self._username, self._password),
            proxies={"http": None, "https": None},  # type: ignore
        )

    @property
    def config_filename(self) -> str:
        config_path = self._http_get(EverEndpoints.config_path).text

        return Path(config_path).name

    def get_runtime(self) -> int:
        if self._start_time is None:
            self._start_time = int(self._http_get(EverEndpoints.start_time).text)

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
            except ConnectionClosedError:
                logger.debug("Connetion closed by server")
            except Exception:
                logger.debug(traceback.format_exc())

        monitor_thread = ErtThread(
            name="everest_gui_event_monitor",
            target=passthrough_ws_events,
            daemon=True,
        )

        return event_queue, monitor_thread

    def create_run_model_api(self) -> RunModelAPI:
        def start_fn(
            evaluator_server_config: EvaluatorServerConfig,
            rerun_failed_realizations: bool = False,
        ) -> None:
            pass

        return RunModelAPI(
            experiment_name=self.config_filename,
            supports_rerunning_failed_realizations=False,
            start_simulations_thread=start_fn,
            cancel=self.stop,
            get_runtime=self.get_runtime,
            has_failed_realizations=lambda: False,
        )

    def stop(self) -> None:
        try:
            response = self._http_post(EverEndpoints.stop)

            if response.status_code == 200:
                logger.info("Cancelled experiment from Everest")
                print("Successfully cancelled experiment")
            else:
                logger.error(
                    f"Failed to cancel Everest experiment: "
                    f"POST @ {self._url}/{EverEndpoints.stop}, "
                    f"server responded with status {response.status_code}: "
                    f"{HTTPStatus(response.status_code).phrase}"
                )
                print("Failed to cancel experiment")

        except requests.exceptions.ConnectionError as e:
            logger.error(
                "Connection error when cancelling Everest "
                f"experiment: {''.join(traceback.format_exception(e))}"
            )
            print("Failed to cancel experiment")

        except HTTPError as e:
            logger.error(
                "HTTP error when cancelling Everest "
                f"experiment: {''.join(traceback.format_exception(e))}"
            )
            print("Failed to cancel experiment")
