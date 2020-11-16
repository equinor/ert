import json

import pytest
from ert_shared.storage import ERT_STORAGE
from ert_shared.storage.http_server import FlaskWrapper
from tests.storage import api, db_api, populated_database, initialize_databases


@pytest.fixture()
def test_client(db_api):
    # Flask provides a way to test your application by exposing the Werkzeug test Client
    # and handling the context locals for you.
    flWrapper = FlaskWrapper(secure=False, url=ERT_STORAGE.SQLALCHEMY_URL)
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
                assert response["data"] is not None

        for response in ensemble["responses"]:
            assert response["ref_url"] is not None


def test_observation(test_client):
    resp = test_client.get("/ensembles/1")
    ens = json.loads(resp.data)
    expected = {
        "active_mask": [True, False],
        "data_indexes": [2, 3],
        "key_indexes": [0, 3],
        "std": [1, 3],
        "values": [10.1, 10.2],
    }

    resp_url = ens["responses"][0]["ref_url"]
    resp_data = test_client.get(resp_url).data.decode()
    resp = json.loads(resp_data)
    observations = resp["observations"]

    actual = {}
    for obs in observations:
        for name, data_def in obs["data"].items():
            data = data_def["data"]
            actual[name] = data

    assert actual == expected


def test_get_single_observation(test_client):
    resp = test_client.get("/observation/observation_one")
    obs = json.loads(resp.data)

    assert obs["attributes"] == {"region": "1"}
    assert obs["name"] == "observation_one"

    key_indexes = obs["data"]["key_indexes"]["data"]
    assert key_indexes == [0, 3]

    data_indexes = obs["data"]["data_indexes"]["data"]
    assert data_indexes == [2, 3]

    values = obs["data"]["values"]["data"]
    assert values == [10.1, 10.2]

    stds = obs["data"]["std"]["data"]
    assert stds == [1, 3]


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
