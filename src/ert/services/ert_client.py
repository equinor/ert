from __future__ import annotations

import io
import json
import logging
import queue
import ssl
import threading
import time
import traceback
from base64 import b64encode
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from functools import wraps
from os import PathLike
from typing import Any, Concatenate, cast
from urllib.parse import quote

import httpx
import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import ValidationError
from websockets.exceptions import ConnectionClosedError
from websockets.sync.client import connect

from _ert.threading import ErtThread
from ert.run_models.event import (
    StatusEvents,
    status_event_from_json,
)
from everest.strings import EverEndpoints

from .shared_client import ErtClientConnectionInfo, Methods, SharedClient

DEFAULT_TIMEOUT = 120
DEFAULT_CACHE_SIZE = 256

_PARQUET = {"accept": "application/x-parquet"}
_EXPERIMENT_SERVER = "/experiment_server"

logger = logging.getLogger(__name__)


def _escape(value: str) -> str:
    """Keys may contain slashes, and the server decodes the path segment once."""
    return quote(quote(value, safe=""))


def _uncached_copy[T](value: T) -> T:
    if isinstance(value, pd.DataFrame):
        return cast("T", value.copy())
    return deepcopy(value)


def _cached[**P, T](
    method: Callable[Concatenate[ErtClient, P], T],
) -> Callable[Concatenate[ErtClient, P], T]:
    """Memoize a method whose result cannot change while the server is up.

    Only apply this to endpoints serving data that is immutable once written.
    Responses grow while an experiment runs, so they must not be cached.
    """

    @wraps(method)
    def wrapper(self: ErtClient, /, *args: P.args, **kwargs: P.kwargs) -> T:
        key = (method.__name__, args, tuple(sorted(kwargs.items())))
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return _uncached_copy(self._cache[key])

        # Fetch outside the lock so that concurrent requests are not serialized.
        value = method(self, *args, **kwargs)

        with self._cache_lock:
            self._cache[key] = value
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return _uncached_copy(value)

    return wrapper


def _filter_params(filter_on: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if filter_on is None else {"filter_on": json.dumps(filter_on)}


def _checked(response: httpx.Response) -> httpx.Response:
    if response.status_code == httpx.codes.UNAUTHORIZED:
        raise httpx.RequestError(message=f"{response.text}")
    if response.status_code != httpx.codes.OK:
        raise httpx.RequestError(
            f" Please report this error and try restarting the application."
            f"{response.text} from url: {response.url}."
        )
    return response


def _response_to_parquet(response: httpx.Response) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(response.content))


class ErtClient:
    """Endpoint-level client for the ERT storage server."""

    def __init__(
        self,
        client: SharedClient,
        timeout: int = DEFAULT_TIMEOUT,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        self._client = client
        self._timeout = timeout
        self._cache_size = cache_size
        self._cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._cache_lock = threading.Lock()

    @classmethod
    def get_client(
        cls,
        project: PathLike[str],
        connect_timeout: int | None = None,
        request_timeout: int = DEFAULT_TIMEOUT,
    ) -> ErtClient:
        """Initialize and connect a client to `project`

        Args:
            project (PathLike[str]): Path to the project directory.
            connect_timeout (int | None, optional): Timeout for establishing connection.
            request_timeout (int, optional): Timeout for requests.

        Returns:
            ErtClient: An instance of the ErtClient connected to the specified project.
        """
        return cls(
            SharedClient.get_client(project, connect_timeout), timeout=request_timeout
        )

    @property
    def client(self) -> SharedClient:
        return self._client

    @property
    def conn_info(self) -> ErtClientConnectionInfo:
        return self._client.conn_info

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    # <-------------- Dark Storage -------------->

    def healthcheck(self) -> str:
        return str(self._get("/healthcheck").json())

    def version(self) -> str:
        return str(self._get("/version").json())

    def experiments(self) -> list[dict[str, Any]]:
        return self._get("/experiments").json()

    def ensemble(self, ensemble_id: str) -> dict[str, Any]:
        return dict(self._get(f"/ensembles/{ensemble_id}").json())

    def ensemble_blobs(self, ensemble_id: str) -> list[dict[str, Any]]:
        return list(self._get(f"/ensembles/{ensemble_id}/blobs").json())

    def ensemble_blob(self, ensemble_id: str, uri: str) -> bytes:
        return self._get(f"/ensembles/{ensemble_id}/blobs/{_escape(uri)}").content

    def parameter(self, ensemble_id: str, parameter_key: str) -> pd.DataFrame:
        return self._parameter(ensemble_id, parameter_key)

    @_cached
    def _parameter(self, ensemble_id: str, parameter_key: str) -> pd.DataFrame:
        return _response_to_parquet(
            self._get(
                f"/ensembles/{ensemble_id}/parameters/{_escape(parameter_key)}",
                headers=_PARQUET,
            )
        )

    def parameter_std_dev(
        self, ensemble_id: str, parameter_key: str, z: int
    ) -> npt.NDArray[np.float32]:
        response = self._request(
            "GET",
            f"/ensembles/{ensemble_id}/parameters/{_escape(parameter_key)}/std_dev",
            params={"z": z},
        )
        if response.status_code != httpx.codes.OK:
            return np.array([])
        return np.load(io.BytesIO(response.content))

    def ert_response(
        self,
        ensemble_id: str,
        response_key: str,
        filter_on: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        return _response_to_parquet(
            self._get(
                f"/ensembles/{ensemble_id}/responses/{_escape(response_key)}",
                headers=_PARQUET,
                params=_filter_params(filter_on),
            )
        )

    def gradient(self, ensemble_id: str, response_key: str) -> pd.DataFrame:
        return self._gradient(ensemble_id, response_key)

    @_cached
    def _gradient(self, ensemble_id: str, response_key: str) -> pd.DataFrame:
        return _response_to_parquet(
            self._request(
                "GET",
                f"/ensembles/{ensemble_id}/gradients/{_escape(response_key)}",
                headers=_PARQUET,
            )
        )

    @_cached
    def experiment_observations(self, experiment_id: str) -> list[dict[str, Any]]:
        return list(self._get(f"/experiments/{experiment_id}/observations").json())

    def response_observations(
        self,
        ensemble_id: str,
        response_key: str,
        filter_on: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return list(
            self._get(
                f"/ensembles/{ensemble_id}/responses/"
                f"{_escape(response_key)}/observations",
                params=_filter_params(filter_on),
            ).json()
        )

    # <------------- Experiment Server ------------->

    def experiment_server_is_running(self) -> bool:
        try:
            response = self._request("GET", f"{_EXPERIMENT_SERVER}/", auth=self._auth)
        except httpx.TransportError:
            return False
        return response.status_code == httpx.codes.OK

    def experiment_ids(self) -> list[str]:
        response = self._experiment_server_get(EverEndpoints.EXPERIMENTS)
        return list(response.json()["experiment_ids"])

    def experiment_status(self, experiment_id: str) -> dict[str, Any]:
        return dict(
            self._experiment_server_get(
                f"{EverEndpoints.STATUS}/{experiment_id}"
            ).json()
        )

    def experiment_config(self, experiment_id: str) -> dict[str, str]:
        return self._experiment_server_get(
            f"{EverEndpoints.CONFIG_PATH}/{experiment_id}"
        ).json()

    def experiment_start_time(self, experiment_id: str) -> int:
        return int(
            self._experiment_server_get(
                f"{EverEndpoints.START_TIME}/{experiment_id}"
            ).text
        )

    def setup_event_queue_from_ws_endpoint(
        self,
        experiment_id: str,
        refresh_interval: float = 0.01,
        open_timeout: float = 30,
        websocket_recv_timeout: float = 1.0,
    ) -> tuple[queue.SimpleQueue[StatusEvents], ErtThread]:
        """Return a queue of experiment events and the thread that fills it.

        The caller owns the thread and must start it.
        """
        event_queue: queue.SimpleQueue[StatusEvents] = queue.SimpleQueue()

        url = (
            self.conn_info.base_url.replace("https://", "wss://")
            + f"{_EXPERIMENT_SERVER}/{EverEndpoints.EVENTS}/{experiment_id}"
        )
        username, password = self._auth
        credentials = b64encode(f"{username}:{password}".encode()).decode()

        def passthrough_ws_events() -> None:
            try:  # ruff: ignore[too-many-statements-in-try-clause]
                with connect(
                    url,
                    ssl=self._ssl_context,
                    open_timeout=open_timeout,
                    additional_headers={"Authorization": f"Basic {credentials}"},
                ) as websocket:
                    while True:
                        try:
                            message = websocket.recv(timeout=websocket_recv_timeout)
                        except TimeoutError:
                            message = None
                        if message:
                            try:
                                event_queue.put(status_event_from_json(message))
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
            name="ert_storage_api_event_monitor",
            target=passthrough_ws_events,
            daemon=True,
        )

        return event_queue, monitor_thread

    def start_experiment(self, config: dict[str, Any]) -> str:
        response = self._request(
            "POST",
            f"{_EXPERIMENT_SERVER}/{EverEndpoints.START_EXPERIMENT}",
            auth=self._auth,
            json=config,
        )
        return str(_checked(response).json()["experiment_id"])

    def stop_experiment_server(self) -> bool:
        return (
            self._request(
                "POST",
                f"{_EXPERIMENT_SERVER}/{EverEndpoints.STOP}",
                auth=self._auth,
            ).status_code
            == 200
        )

    def runpath_exists(self, paths: list[str]) -> bool:
        response = self._request(
            "POST",
            f"{_EXPERIMENT_SERVER}/{EverEndpoints.RUNPATH}",
            auth=self._auth,
            json={"paths": paths},
        )
        return response.status_code == httpx.codes.OK

    # <-------------- Internals -------------->

    @property
    def _ssl_context(self) -> ssl.SSLContext | None:
        cert = self._client.conn_info.cert
        if not isinstance(cert, str):
            return None
        return ssl.create_default_context(cafile=cert)

    @property
    def _auth(self) -> tuple[str, str]:
        """Experiment-server routes authenticate with HTTP Basic, not the token
        header the storage routes use.
        """
        token = self._client.conn_info.auth_token
        if token is None:
            raise RuntimeError("No authentication token found in storage session")
        return ("username", token)

    def _experiment_server_get(self, path: str) -> httpx.Response:
        return _checked(
            self._request("GET", f"{_EXPERIMENT_SERVER}/{path}", auth=self._auth)
        )

    def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        return _checked(self._request("GET", url, **kwargs))

    def _request(self, method: Methods, url: str, **kwargs: Any) -> httpx.Response:
        kwargs.setdefault("timeout", self._timeout)
        return self._client.request(method, url, **kwargs)
