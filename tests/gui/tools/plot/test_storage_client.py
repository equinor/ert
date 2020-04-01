
from flask import Flask, Response
import json
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
    # Not yet implemented
    pass
    #with patch("ert_gui.tools.plot.storage_client.requests", test_client):
    #    api = StorageClient(base_url="")
    #    result = api.observations_for_obs_keys(
    #        case="ensemble_name", obs_keys=["observation_one"]
    #    )
#
    #    idx = pd.MultiIndex.from_arrays(
    #        [[2, 3], [0, 3]], names=["data_index", "key_index"]
    #    )
    #    expected = pd.DataFrame({"OBS": [10.1, 10.2], "STD": [1, 3]}, index=idx).T
#
    #    assert result.equals(expected)


def test_response_values(test_client):
    with patch("ert_gui.tools.plot.storage_client.requests", test_client):
        api = StorageClient(base_url="")
        result = api.data_for_key(case="ensemble_name", key="response_one")

        idx = pd.MultiIndex.from_arrays(
            [["response_one", "response_one"], [0, 1]], names=["key", "index"]
        )
        expected = pd.DataFrame([[11.1, 11.1], [11.2, 11.2]], index=idx).T

        pd.testing.assert_frame_equal(result, expected)
