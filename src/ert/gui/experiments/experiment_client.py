from __future__ import annotations

import logging
import queue
import ssl
import time
import traceback
from base64 import b64encode
from http import HTTPStatus
from typing import Any

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


class ExperimentClient:
    def __init__(
        self,
        url: str,
        cert_file: str,
        username: str,
        password: str,
        ssl_context: ssl.SSLContext,
        run_id: str | None = None,
    ) -> None:
        self._run_id = run_id
        self._url = url
        self._cert = cert_file
        self._username = username
        self._password = password
        self._ssl_context = ssl_context

        self._is_alive = False
        self._start_time: int | None = None

    def _http_get(self, endpoint: str) -> requests.Response:
        return requests.get(
            f"{self._url}/{endpoint}",
            verify=self._cert,
            auth=(self._username, self._password),
            proxies={"http": None, "https": None},  # type: ignore
            timeout=(5, 30),
        )

    def _http_post(self, endpoint: str) -> requests.Response:
        return requests.post(
            f"{self._url}/{endpoint}",
            verify=self._cert,
            auth=(self._username, self._password),
            proxies={"http": None, "https": None},  # type: ignore
            timeout=(5, 30),
        )

    def _http_post_json(self, endpoint: str, body: dict[str, Any]) -> requests.Response:
        return requests.post(
            f"{self._url}/{endpoint}",
            json=body,
            verify=self._cert,
            auth=(self._username, self._password),
            proxies={"http": None, "https": None},  # type: ignore
            timeout=(5, 30),
        )

    @property
    def config(self) -> dict[str, str]:
        return self._http_get(f"{EverEndpoints.config_path}/{self._run_id}").json()

    @property
    def credentials(self) -> str:
        return b64encode(f"{self._username}:{self._password}".encode()).decode()

    def check_runpath(self, config_json: dict[str, Any]) -> dict[str, Any]:
        response = self._http_post_json(EverEndpoints.check_runpath, config_json)
        response.raise_for_status()
        return response.json()

    def delete_runpaths(self, config_json: dict[str, Any]) -> None:
        response = self._http_post_json(EverEndpoints.delete_runpaths, config_json)
        response.raise_for_status()

    def start_experiment(self, config_json: dict[str, Any]) -> tuple[str, bool]:
        response = self._http_post_json(EverEndpoints.start_experiment, config_json)
        response.raise_for_status()
        data = response.json()
        self._run_id = data["run_id"]
        return self._run_id, data.get("supports_rerunning_failed_realizations", False)

    def rerun_failed(self) -> tuple[str, bool]:
        assert self._run_id is not None, "No active run to rerun"
        response = self._http_post(
            f"{EverEndpoints.start_experiment}?rerun_from_run_id={self._run_id}"
        )
        response.raise_for_status()
        data = response.json()
        self._run_id = data["run_id"]
        return self._run_id, data.get("supports_rerunning_failed_realizations", False)

    def has_failed_realizations(self) -> bool:
        assert self._run_id is not None, "No active run"
        response = self._http_get(
            f"{EverEndpoints.has_failed_realizations}/{self._run_id}"
        )
        response.raise_for_status()
        return bool(response.json().get("has_failed", False))

    def failed_realizations_mask(self) -> list[bool]:
        assert self._run_id is not None, "No active run"
        response = self._http_get(
            f"{EverEndpoints.failed_realizations_mask}/{self._run_id}"
        )
        response.raise_for_status()
        return [bool(v) for v in response.json().get("mask", [])]

    def setup_event_queue_from_ws_endpoint(
        self,
        event_queue: queue.SimpleQueue[StatusEvents] | None = None,
        refresh_interval: float = 0.01,
        open_timeout: float = 30,
        websocket_recv_timeout: float = 1.0,
    ) -> tuple[queue.SimpleQueue[StatusEvents], ErtThread]:
        if event_queue is None:
            event_queue = queue.SimpleQueue()
        out_queue = event_queue

        def passthrough_ws_events() -> None:
            assert self._run_id is not None, "No active run to stream events for"
            try:
                with connect(
                    self._url.replace("https://", "wss://")
                    + f"/{EverEndpoints.events}/{self._run_id}",
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
                                out_queue.put(event)
                            except ValidationError as e:
                                logger.error(
                                    "Error when processing event %s", exc_info=e
                                )

                        time.sleep(refresh_interval)
            except ConnectionClosedError:
                logger.debug("Connection closed by server")
            except Exception:
                logger.debug(traceback.format_exc())

        monitor_thread = ErtThread(
            name="ert_gui_event_monitor",
            target=passthrough_ws_events,
            daemon=True,
        )

        return out_queue, monitor_thread

    def create_run_model_api(
        self,
        experiment_name: str,
        supports_rerunning_failed_realizations: bool,
    ) -> RunModelAPI:
        def start_fn(
            evaluator_server_config: EvaluatorServerConfig,
            rerun_failed_realizations: bool = False,
        ) -> None:
            pass

        return RunModelAPI(
            experiment_name=experiment_name,
            supports_rerunning_failed_realizations=supports_rerunning_failed_realizations,
            start_simulations_thread=start_fn,
            cancel=self.stop,
            has_failed_realizations=self.has_failed_realizations,
        )

    def stop(self) -> None:
        try:
            response = self._http_post(EverEndpoints.stop)

            if response.status_code == 200:
                logger.info("Cancelled experiment from EVEREST")
                print("Successfully cancelled experiment")
            else:
                logger.error(
                    f"Failed to cancel EVEREST experiment: "
                    f"POST @ {self._url}/{EverEndpoints.stop}, "
                    f"server responded with status {response.status_code}: "
                    f"{HTTPStatus(response.status_code).phrase}"
                )
                print("Failed to cancel experiment")

        except requests.exceptions.ConnectionError as e:
            logger.error(
                "Connection error when cancelling EVEREST "
                f"experiment: {''.join(traceback.format_exception(e))}"
            )
            print("Failed to cancel experiment")

        except HTTPError as e:
            logger.error(
                "HTTP error when cancelling EVEREST "
                f"experiment: {''.join(traceback.format_exception(e))}"
            )
            print("Failed to cancel experiment")
