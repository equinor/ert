from datetime import datetime

import pandas as pd
import pytest
from ert_shared.storage import ERT_STORAGE
from ert_shared.storage.client import StorageClient
from ert_shared.storage.server_monitor import ServerMonitor
from tests.storage import api, db_api, populated_database, initialize_databases


@pytest.fixture
def server(db_api, request):
    proc = ServerMonitor(rdb_url=ERT_STORAGE.sqlalchemy_url, lockfile=False)
    proc.start()
    request.addfinalizer(lambda: proc.shutdown())
    yield proc


@pytest.fixture
def storage_client(server):
    yield StorageClient(server.fetch_url(), server.fetch_auth())


def test_all_keys(storage_client):
    names = {key["key"] for key in storage_client.all_data_type_keys()}
    assert names == {"response_one", "response_two", "G:A", "G:B", "group:key1"}


def test_observation_values(storage_client):
    # Data refs are collected during the call to <all_data_type_keys>
    response_key = [
        key
        for key in storage_client.all_data_type_keys()
        if key["key"] == "response_one"
    ][0]

    result = storage_client.observations_for_obs_keys(
        case="ensemble_name", obs_keys=response_key["observations"]
    )

    idx = pd.MultiIndex.from_arrays(
        [["response_one", "response_one"], [0, 3], [2, 3]],
        names=["obs_key", "key_index", "data_index"],
    )
    expected = pd.DataFrame({"OBS": [10.1, 10.2], "STD": [1, 3]}, index=idx).T

    pd.testing.assert_frame_equal(result, expected)

    # Response two has datetime indexes and consists of two individual observations put together
    response_key = [
        key
        for key in storage_client.all_data_type_keys()
        if key["key"] == "response_two"
    ][0]

    result = storage_client.observations_for_obs_keys(
        case="ensemble_name", obs_keys=response_key["observations"]
    )

    format = "%Y-%m-%d %H:%M:%S"
    idx = pd.MultiIndex.from_arrays(
        [
            ["response_two", "response_two"],
            [
                datetime.strptime("2000-01-01 20:01:01", format),
                datetime.strptime("2000-01-02 20:01:01", format),
            ],
            [4, 5],
        ],
        names=["obs_key", "key_index", "data_index"],
    )
    expected = pd.DataFrame({"OBS": [10.3, 10.4], "STD": [2, 2.5]}, index=idx).T

    pd.testing.assert_frame_equal(result, expected)


def test_response_values(storage_client):
    result = storage_client.data_for_key(case="ensemble_name", key="response_one")

    idx = [3, 5, 8, 9]

    expected = pd.DataFrame(
        [[11.1, 11.1], [11.2, 11.2], [9.9, 9.9], [9.3, 9.3]], index=idx
    ).T

    pd.testing.assert_frame_equal(result, expected)

    result = storage_client.data_for_key(case="ensemble_name", key="response_two")

    format = "%Y-%m-%d %H:%M:%S"
    idx = [
        datetime.strptime("2000-01-01 20:01:01", format),
        datetime.strptime("2000-01-02 20:01:01", format),
        datetime.strptime("2000-01-02 20:01:01", format),
        datetime.strptime("2000-01-02 20:01:01", format),
        datetime.strptime("2000-01-02 20:01:01", format),
        datetime.strptime("2000-01-02 20:01:01", format),
    ]

    expected = pd.DataFrame(
        [
            [12.1, 12.1],
            [12.2, 12.2],
            [11.1, 11.1],
            [11.2, 11.2],
            [9.9, 9.9],
            [9.3, 9.3],
        ],
        index=idx,
    ).T

    pd.testing.assert_frame_equal(result, expected)
