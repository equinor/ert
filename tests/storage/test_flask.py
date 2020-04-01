import json
import pprint

import flask
import pytest
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.http_server import FlaskWrapper
from ert_shared.storage.rdb_api import RdbApi
from flask import Response, request

from tests.storage import db_session, engine, populated_db, tables


@pytest.fixture()
def test_client(populated_db):
    # Flask provides a way to test your application by exposing the Werkzeug test Client
    # and handling the context locals for you.
    rdb_api = RdbApi(populated_db)
    blob_api = BlobApi(populated_db)
    flWrapper = FlaskWrapper(rdb_api=rdb_api, blob_api=blob_api)
    testing_client = flWrapper.app.test_client()
    # Establish an application context before running the tests.
    ctx = flWrapper.app.app_context()
    ctx.push()
    yield testing_client
    ctx.pop()


def test_api(test_client):
    response = test_client.get("/ensembles")
    print(response.data)
    print(response.mimetype)
    ensembles = json.loads(response.data)

    for ens in ensembles["ensembles"]:
        print("########## ENSEMBLE #############")
        url = ens["ref_url"]
        ensemble = json.loads(test_client.get(url).data)
        pprint.pprint(ensemble)

        for real in ensemble["realizations"]:
            print("########## ENSEMBLE - realization #############")
            realization = json.loads(test_client.get(real["ref_url"]).data)
            pprint.pprint(realization)

            for response in realization["responses"]:
                print("########## ENSEMBLE - realization - response #############")
                response_data = test_client.get(response["data_ref"])
                print(response_data.data)

        for response in ensemble["responses"]:
            print("########## ENSEMBLE - response #############")
            response_data = test_client.get(response["ref_url"])
            pprint.pprint(json.loads(response_data.data))


def test_observation(test_client):
    resp = test_client.get("/ensembles/1")
    ens = json.loads(resp.data)
    expected = {
        ("data_indexes", "2,3"),
        ("key_indexes", "0,3"),
        ("std", "1,3"),
        ("values", "10.1,10.2"),
    }

    actual = set()

    resp_url = ens["responses"][0]["ref_url"]
    resp_data = test_client.get(resp_url).data.decode()
    resp = json.loads(resp_data)
    obs = resp["observation"]

    for data_def in obs["data"]:
        url = data_def["data_ref"]
        name = data_def["name"]
        resp = test_client.get(url)
        actual.add((name, resp.data.decode("utf-8")))


    assert actual == expected
