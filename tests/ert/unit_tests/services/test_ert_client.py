import io
import logging
from typing import Any
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest
from websockets.exceptions import ConnectionClosedError, InvalidState
from websockets.frames import Close

from ert.ensemble_evaluator import EndEvent
from ert.services.ert_client import ErtClient
from ert.services.shared_client import SharedClient


class RecordingResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = ""
        self.url = ""

    def json(self) -> Any:
        return self._payload

    @property
    def content(self) -> bytes:
        if isinstance(self._payload, bytes):
            return self._payload
        raise AssertionError("payload is not binary")


class RecordingClient:
    """Counts requests per url so tests can tell a cache hit from a refetch."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    def request(self, method: str, url: str, **kwargs: Any) -> RecordingResponse:
        self.requests.append(url)
        return RecordingResponse(self._payload_for(url))

    def count_requests(self, fragment: str) -> int:
        return len([url for url in self.requests if fragment in url])

    @staticmethod
    def _payload_for(url: str) -> Any:
        if "/parameters/" in url or "/responses/" in url or "/gradients/" in url:
            stream = io.BytesIO()
            pd.DataFrame({"0": [1.0, 2.0, 3.0]}).to_parquet(stream)
            return stream.getvalue()
        if url == "/experiments":
            return [{"id": "exp_1", "ensemble_ids": ["ens_1"]}]
        return {"userdata": {"name": "ensemble"}}


@pytest.fixture
def client() -> RecordingClient:
    return RecordingClient()


@pytest.fixture
def api(client: RecordingClient) -> ErtClient:
    return ErtClient(client)  # type: ignore


def test_that_repeated_parameter_calls_issue_a_single_request(api, client):
    for _ in range(5):
        api.parameter("1", "2")

    assert client.count_requests("/parameter") == 1


def test_that_experiments_are_cached_separately_per_experiment_id(api, client):
    api.experiment_observations("1")
    api.experiment_observations("1")
    api.experiment_observations("2")

    assert client.count_requests("/experiments/1/observations") == 1
    assert client.count_requests("/experiments/2/observations") == 1


def test_that_ert_response_calls_always_reach_the_server(api, client):
    for _ in range(3):
        api.ert_response("ens_1", "FOPR")

    assert client.count_requests("/responses/") == 3


def test_that_ert_response_observations_always_reach_the_server(api, client):
    for _ in range(3):
        api.response_observations("ens_1", "FOPR")

    assert client.count_requests("/ensembles/ens_1/response") == 3


def test_that_clear_cache_makes_the_next_call_refetch(api, client):
    api.gradient("1", "2")  # Add to cache
    api.gradient("1", "2")  # Cache hit, request is never made
    api.clear_cache()  # Cache cleared
    api.gradient("1", "2")  # Cache miss, request is made again

    assert client.count_requests("/ensembles/1/gradients/") == 2


def test_that_least_recently_used_entries_are_evicted_beyond_cache_size(client):
    api = ErtClient(client, cache_size=2)

    api.experiment_observations("ex_1")  # (1, )
    api.experiment_observations("ex_2")  # (2, 1)
    api.experiment_observations("ex_1")  # (1, 2)
    api.experiment_observations("ex_3")  # (3, 1)
    api.experiment_observations("ex_1")  # (1, 3)
    api.experiment_observations("ex_2")  # (1, 2)

    assert (
        client.count_requests("/experiments/ex_1/observations") == 1
    )  # Always in cache after first request
    assert (
        client.count_requests("/experiments/ex_2/observations") == 2
    )  # Evicted before second request.


def test_that_mutating_a_returned_parameter_frame_leaves_the_cache_intact(api, client):
    frame = api.parameter("ens_1", "gen_kw")

    # Assert that request was cached
    assert ("_parameter", ("ens_1", "gen_kw"), ()) in api._cache

    # Mutate the returned frame
    frame[0, 0] = -50.0

    # Assert that the cached value is not mutated.
    assert api.parameter("ens_1", "gen_kw")["0"].to_list() == [1.0, 2.0, 3.0]


@pytest.fixture
def event_client(monkeypatch):
    transport = MagicMock(spec=SharedClient)
    transport.conn_info.base_url = "https://localhost:1234"
    transport.conn_info.auth_token = "token"
    transport.conn_info.cert = False
    connect = MagicMock()
    connection = connect.return_value
    monkeypatch.setattr("ert.services.ert_client.connect", connect)
    return ErtClient(transport), connection, connect


def test_that_closing_event_iterator_releases_connection(event_client):
    api, connection, _ = event_client
    event = EndEvent(failed=False, msg="completed")
    connection.recv.return_value = event.model_dump_json()
    events = api.iter_events("experiment")

    assert next(events) == event
    connection.close.assert_not_called()
    events.close()
    connection.close.assert_called_once()


def test_that_failure_to_close_websocket_does_not_mask_generator_exit(event_client):
    api, connection, _ = event_client
    event = EndEvent(failed=False, msg="completed")
    connection.recv.return_value = event.model_dump_json()
    connection.close.side_effect = InvalidState("connection is closing")
    events = api.iter_events("experiment")

    assert next(events) == event
    events.close()


def test_that_abnormal_event_stream_closure_logs_experiment_and_close_reason(
    event_client, caplog
):
    caplog.set_level(logging.INFO, logger="ert.services.ert_client")
    api, connection, _ = event_client
    connection.recv.side_effect = ConnectionClosedError(
        Close(1008, "unknown experiment"), Close(1008, "unknown experiment"), True
    )

    assert list(api.iter_events("experiment")) == []

    assert "WebSocket event stream for experiment experiment" in caplog.text
    assert "closed abnormally" in caplog.text
    assert "1008" in caplog.text
    assert "unknown experiment" in caplog.text
    assert "Connected to WebSocket event stream" in caplog.text
    assert "ended after 0 events" in caplog.text
    connection.close.assert_called_once()


def test_that_event_stream_logs_first_event_and_total_on_close(event_client, caplog):
    caplog.set_level(logging.INFO, logger="ert.services.ert_client")
    api, connection, _ = event_client
    event = EndEvent(failed=False, msg="completed")
    connection.recv.return_value = event.model_dump_json()
    events = api.iter_events("experiment")

    assert next(events) == event
    events.close()

    assert "Received first WebSocket event for experiment experiment: EndEvent" in (
        caplog.text
    )
    assert "ended after 1 events" in caplog.text


@pytest.mark.parametrize("status_code", [200, 401, 503])
@pytest.mark.parametrize("timeout", [None, 1.0])
def test_that_server_probe_checks_authenticated_endpoint_with_requested_timeout(
    event_client, status_code, timeout
):
    api, _, _ = event_client
    api.client.request.return_value.status_code = status_code

    assert api.server_is_running(timeout=timeout) is (status_code == 200)

    api.client.request.assert_called_once_with(
        "GET",
        "/experiment_server/",
        auth=("username", "token"),
        timeout=120 if timeout is None else timeout,
    )


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
def test_that_server_probe_returns_false_on_transport_error(event_client, error_type):
    api, _, _ = event_client
    api.client.request.side_effect = error_type("unavailable")

    assert not api.server_is_running(timeout=1)
