import json

import flask
import pytest
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.http_server import FlaskWrapper
from ert_shared.storage.rdb_api import RdbApi
from flask import Response, request

from tests.storage import db_info


@pytest.fixture()
def test_client(db_info):
    populated_db, _ = db_info
    # Flask provides a way to test your application by exposing the Werkzeug test Client
    # and handling the context locals for you.
    flWrapper = FlaskWrapper(rdb_url=populated_db, blob_url=populated_db, secure=False)
    testing_client = flWrapper.app.test_client()
    # Establish an application context before running the tests.
    with flWrapper.app.app_context():
        yield testing_client


def test_api(test_client):
    response = test_client.get("/ensembles")
    ensembles = json.loads(response.data)

    for ens in ensembles["ensembles"]:
        url = ens["ref_url"]
        ensemble = json.loads(test_client.get(url).data)

        for real in ensemble["realizations"]:
            realization = json.loads(test_client.get(real["ref_url"]).data)

            for response in realization["responses"]:
                response_data = test_client.get(response["data_url"])

        for response in ensemble["responses"]:
            response_data = test_client.get(response["ref_url"])


def test_observation(test_client):
    resp = test_client.get("/ensembles/1")
    ens = json.loads(resp.data)
    expected = {
        ("active_mask", "True,False"),
        ("data_indexes", "2,3"),
        ("key_indexes", "0,3"),
        ("std", "1,3"),
        ("values", "10.1,10.2"),
    }

    actual = set()

    resp_url = ens["responses"][0]["ref_url"]
    resp_data = test_client.get(resp_url).data.decode()
    resp = json.loads(resp_data)
    observations = resp["observations"]

    for obs in observations:
        for name, data_def in obs["data"].items():
            url = data_def["data_url"]
            resp = test_client.get(url)
            actual.add((name, resp.data.decode("utf-8")))

    assert actual == expected


def test_get_single_observation(test_client):
    resp = test_client.get("/observation/observation_one")
    obs = json.loads(resp.data)

    assert obs["attributes"] == {"region": "1"}
    assert obs["name"] == "observation_one"

    key_indexes_url = obs["data"]["key_indexes"]["data_url"]
    key_indexes = test_client.get(key_indexes_url).data
    assert key_indexes == b"0,3"

    data_indexes_url = obs["data"]["data_indexes"]["data_url"]
    data_indexes = test_client.get(data_indexes_url).data
    assert data_indexes == b"2,3"

    values_url = obs["data"]["values"]["data_url"]
    values = test_client.get(values_url).data
    assert values == b"10.1,10.2"

    stds_url = obs["data"]["std"]["data_url"]
    stds = test_client.get(stds_url).data
    assert stds == b"1,3"


def test_get_ensemble_id_404(test_client):
    resp = test_client.get("/ensembles/not_existing")
    assert resp.status_code == 404


def test_get_single_observation_404(test_client):
    resp = test_client.get("/observation/not_existing")
    assert resp.status_code == 404


def test_get_observation_attributes(test_client):
    create_resp = test_client.post(
        "/observation/observation_one/attributes",
        data="""{"attributes": {"region": "1", "depth": "9000"}}""",
        content_type="application/json",
    )
    assert create_resp.status_code == 201, create_resp.status
    resp = test_client.get("/observation/observation_one/attributes")
    obs = json.loads(resp.data)
    expected = {"attributes": {"region": "1", "depth": "9000"}}
    assert obs == expected


def test_parameter(test_client):
    resp = test_client.get("/ensembles/1/parameters/3")
    schema = json.loads(resp.data)
    assert schema["key"] == "key1"
    assert schema["group"] == "group"
    assert schema["prior"]["function"] == "function"


def test_blob_404(test_client):
    resp = test_client.get("/data/notfound")
    assert resp.status_code == 404


def test_get_batched_response(test_client):
    resp_schema = _fetch_response(
        test_client, ensemble_name="ensemble_name", response_name="response_two"
    )
    data_url = resp_schema["alldata_url"]

    data_resp = test_client.get(data_url)

    expected = b"12.1,12.2,11.1,11.2,9.9,9.3\n12.1,12.2,11.1,11.2,9.9,9.3"
    csv = data_resp.data
    assert expected == csv


def test_get_batched_response_missing(test_client):
    data_url = "/ensembles/1/responses/none/data"
    data_resp = test_client.get(data_url)
    assert data_resp.status_code == 404


def test_get_batched_parameter(test_client):
    param_schema = _fetch_parameter(
        test_client,
        ensemble_name="ensemble_name",
        parameter_name="A",
        parameter_group="G",
    )
    data_url = param_schema["alldata_url"]

    data_resp = test_client.get(data_url)

    expected = b"1\n1"
    csv = data_resp.data
    assert expected == csv


def test_get_batched_parameter_missing(test_client):
    data_url = "/ensembles/1/parameters/42/data"
    data_resp = test_client.get(data_url)
    assert data_resp.status_code == 404


def _fetch_ensemble(test_client, ensemble_name):
    ensembles_resp = test_client.get("/ensembles")
    ensembles_schema = json.loads(ensembles_resp.data)

    ensemble_ref = next(
        ens for ens in ensembles_schema["ensembles"] if ens["name"] == ensemble_name
    )
    ensemble_url = ensemble_ref["ref_url"]

    ensemble_schema = json.loads(test_client.get(ensemble_url).data)
    return ensemble_schema


def _fetch_response(test_client, ensemble_name, response_name):
    ensemble_schema = _fetch_ensemble(test_client, ensemble_name)
    response_ref = next(
        resp for resp in ensemble_schema["responses"] if resp["name"] == response_name
    )
    response_url = response_ref["ref_url"]
    response_schema = json.loads(test_client.get(response_url).data)
    return response_schema


def _fetch_parameter(test_client, ensemble_name, parameter_name, parameter_group):
    ensemble_schema = _fetch_ensemble(test_client, ensemble_name)
    response_ref = next(
        param
        for param in ensemble_schema["parameters"]
        if param["key"] == parameter_name and param["group"] == parameter_group
    )
    response_url = response_ref["ref_url"]
    data = test_client.get(response_url).data
    response_schema = json.loads(data)
    return response_schema
