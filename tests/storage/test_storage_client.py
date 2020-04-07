from flask import Flask, Response, request
from datetime import datetime
import pytest
import pandas as pd

from tests.storage import db_info, db_connection, engine, tables
from ert_shared.storage.client import StorageClient
from ert_shared.storage.http_server import FlaskWrapper

import requests
import threading
import socket
import time
import os


@pytest.fixture()
def server_url(db_info):
    _environ = dict(os.environ)
    os.environ["NO_PROXY"] = "localhost"

    populated_db, _ = db_info
    flWrapper = FlaskWrapper(rdb_url=populated_db, blob_url=populated_db)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 0))
    port = sock.getsockname()[1]
    sock.close()

    def run_app():
        flWrapper.app.run(host="0.0.0.0", debug=False, port=port)

    threading.Thread(target=run_app).start()

    url = "http://localhost:{}".format(port)

    for x in range(20):
        time.sleep(0.1)
        try:
            requests.get(url)
            break
        except requests.exceptions.RequestException:
            continue

    yield url

    requests.post(url + "/shutdown")
    os.environ.clear()
    os.environ.update(_environ)


def test_all_keys(server_url):
    api = StorageClient(base_url=server_url)
    names = set([key["key"] for key in api.all_data_type_keys()])
    assert names == set(["response_one", "response_two", "G:A", "G:B", "group:key1"])


def test_observation_values(server_url):
    api = StorageClient(base_url=server_url)
    # Data refs are collected during the call to <all_data_type_keys>
    response_key = [
        key for key in api.all_data_type_keys() if key["key"] == "response_one"
    ][0]

    result = api.observations_for_obs_keys(
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
        key for key in api.all_data_type_keys() if key["key"] == "response_two"
    ][0]

    result = api.observations_for_obs_keys(
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


def test_response_values(server_url):
    api = StorageClient(base_url=server_url)
    result = api.data_for_key(case="ensemble_name", key="response_one")

    idx = pd.MultiIndex.from_arrays(
        [
            ["response_one", "response_one", "response_one", "response_one"],
            [3, 5, 8, 9],
        ],
        names=["key", "index"],
    )
    expected = pd.DataFrame(
        [[11.1, 11.1], [11.2, 11.2], [9.9, 9.9], [9.3, 9.3]], index=idx
    ).T

    pd.testing.assert_frame_equal(result, expected)

    result = api.data_for_key(case="ensemble_name", key="response_two")

    format = "%Y-%m-%d %H:%M:%S"
    idx = pd.MultiIndex.from_arrays(
        [
            [
                "response_two",
                "response_two",
                "response_two",
                "response_two",
                "response_two",
                "response_two",
            ],
            [
                datetime.strptime("2000-01-01 20:01:01", format),
                datetime.strptime("2000-01-02 20:01:01", format),
                datetime.strptime("2000-01-02 20:01:01", format),
                datetime.strptime("2000-01-02 20:01:01", format),
                datetime.strptime("2000-01-02 20:01:01", format),
                datetime.strptime("2000-01-02 20:01:01", format),
            ],
        ],
        names=["key", "index"],
    )

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
