from flask import Flask, Response
import json
from datetime import datetime
import pytest
import pandas as pd
from mock import patch

from ert_gui.tools.plot.storage_client import StorageClient
from tests.storage import populated_db, db_connection, engine, tables
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.http_server import FlaskWrapper
from ert_shared.storage.rdb_api import RdbApi

import requests

class CustomResponse(Response):
    # As expected, a lot of functions from the requests.models.Response
    # has to be created. Subclassing requests.models.Response
    # instead, just ends us up in the same inverted issue..

    # request has the <function json>, while flask Response use <function get_json>
    # Go figure
    def json(self):
        return self.get_json()

    @property
    def content(self):
        # the <flask.Response.data> contains the same byte array as <requests.models.content>
        class mock_content:
            def __init__(self, content):
                self.content = content

            def decode(self, *args):
                return self.content.decode("utf-8")

        return mock_content(self.data)

    @property
    def encoding(self):
        pass

@pytest.fixture()
def test_client(populated_db):

    rdb_api = RdbApi(populated_db)
    blob_api = BlobApi(populated_db)
    flWrapper = FlaskWrapper(rdb_url=populated_db, blob_url=populated_db)

    # Default response object is <class 'flask.wrappers.Response'>.
    # We would really like this to be exactly as requests response though

    # documentation says response class should be set on test_client
    # that does not have an effect though, and it must be set directly on the wrapper
    # suppose we have a different setup than what is standard.
    flWrapper.app.response_class = CustomResponse
    testing_client = flWrapper.app.test_client()

    # This is from documentation, but does not have an effect in our case
    # testing_client.response_class = CustomResponse

    # Establish an application context before running the tests.
    ctx = flWrapper.app.app_context()
    ctx.push()
    yield testing_client
    ctx.pop()


def test_all_keys(test_client):
    with patch("ert_gui.tools.plot.storage_client.requests", test_client):
        api = StorageClient(base_url="")
        names = set([key["key"] for key in api.all_data_type_keys()])

        assert names == set(["response_one", "response_two", "A", "B"])


def test_observation_values(test_client):

    with patch("ert_gui.tools.plot.storage_client.requests", test_client):
        api = StorageClient(base_url="")
        # Data refs are collected during the call to <all_data_type_keys>
        response_key = [key for key in api.all_data_type_keys() if key["key"]=="response_one"][0]

        result = api.observations_for_obs_keys(
            case="ensemble_name", obs_keys=response_key["observations"]
        )

        idx = pd.MultiIndex.from_arrays(
            [["response_one", "response_one"], [0, 3], [2, 3]], names=["obs_key", "key_index", "data_index"]
        )
        expected = pd.DataFrame({"OBS": [10.1, 10.2], "STD": [1, 3]}, index=idx).T

        pd.testing.assert_frame_equal(result, expected)

        # Response two has datetime indexes and consists of two individual observations put together
        response_key = [key for key in api.all_data_type_keys() if key["key"]=="response_two"][0]

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
                [4, 5]
            ],
            names=["obs_key", "key_index", "data_index"],
        )
        expected = pd.DataFrame({"OBS": [10.3, 10.4], "STD": [2, 2.5]}, index=idx).T

        pd.testing.assert_frame_equal(result, expected)


def test_response_values(test_client):
    with patch("ert_gui.tools.plot.storage_client.requests", test_client):
        api = StorageClient(base_url="")
        result = api.data_for_key(case="ensemble_name", key="response_one")

        idx = pd.MultiIndex.from_arrays(
            [["response_one", "response_one"], [3, 5]], names=["key", "index"]
        )
        expected = pd.DataFrame([[11.1, 11.1], [11.2, 11.2]], index=idx).T

        pd.testing.assert_frame_equal(result, expected)

        result = api.data_for_key(case="ensemble_name", key="response_two")

        format = "%Y-%m-%d %H:%M:%S"
        idx = pd.MultiIndex.from_arrays(
            [
                ["response_two", "response_two"],
                [
                    datetime.strptime("2000-01-01 20:01:01", format),
                    datetime.strptime("2000-01-02 20:01:01", format),
                ],
            ],
            names=["key", "index"],
        )

        expected = pd.DataFrame([[12.1, 12.1], [12.2, 12.2]], index=idx).T

        pd.testing.assert_frame_equal(result, expected)
