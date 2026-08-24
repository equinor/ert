import io
from typing import Any

import polars as pl
import pytest

from ert.services.ert_client import ErtClient


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
            pl.DataFrame({"0": [1.0, 2.0, 3.0]}).write_parquet(stream)
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
