import io
import json
import re

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from requests import Response

from ert.dark_storage.common import get_storage_api_version


@pytest.mark.integration_test
def test_get_experiment(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    answer_json = resp.json()
    assert len(answer_json) == 1
    assert "ensemble_ids" in answer_json[0]
    assert len(answer_json[0]["ensemble_ids"]) == 2
    assert "name" in answer_json[0]


@pytest.mark.integration_test
def test_get_storage_api_version(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments/version")
    answer_json = resp.json()

    assert answer_json == get_storage_api_version()

    version_pattern = r"^\d+\.\d+$"

    assert re.match(version_pattern, answer_json), (
        f"Version format is invalid: {answer_json}"
    )


@pytest.mark.integration_test
def test_get_ensemble(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    experiment_json = resp.json()
    assert len(experiment_json) == 1
    assert len(experiment_json[0]["ensemble_ids"]) == 2

    ensemble_id = experiment_json[0]["ensemble_ids"][0]

    resp: Response = dark_storage_client.get(f"/ensembles/{ensemble_id}")
    ensemble_json = resp.json()

    assert ensemble_json["experiment_id"] == experiment_json[0]["id"]
    assert ensemble_json["userdata"]["name"] in {"iter-0", "iter-1"}
    assert ensemble_json["userdata"]["experiment_name"] == experiment_json[0]["name"]


@pytest.mark.integration_test
def test_get_experiment_ensemble(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    experiment_json = resp.json()
    assert len(experiment_json) == 1
    assert len(experiment_json[0]["ensemble_ids"]) == 2

    experiment_id = experiment_json[0]["id"]

    resp: Response = dark_storage_client.get(f"/experiments/{experiment_id}/ensembles")
    ensembles_json = resp.json()

    assert len(ensembles_json) == 2
    assert ensembles_json[0]["experiment_id"] == experiment_json[0]["id"]
    assert ensembles_json[0]["userdata"]["name"] in {"iter-0", "iter-1"}


@pytest.mark.integration_test
def test_get_responses_with_observations(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    experiment_json = resp.json()[0]

    assert experiment_json["observations"] == {"gen_data": {"POLY_RES": ["POLY_OBS"]}}
    assert experiment_json["responses"] == {
        "gen_data": [
            {
                "response_type": "gen_data",
                "response_key": "POLY_RES",
                "filter_on": {"report_step": [0]},
                "finalized": True,
            }
        ]
    }


@pytest.mark.integration_test
def test_get_response(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    experiment_json = resp.json()

    assert len(experiment_json[0]["ensemble_ids"]) == 2, experiment_json

    ensemble_id1 = experiment_json[0]["ensemble_ids"][0]
    ensemble_id2 = experiment_json[0]["ensemble_ids"][1]

    # Make sure the order is correct
    resp: Response = dark_storage_client.get(f"/ensembles/{ensemble_id1}")
    if resp.json()["userdata"]["name"] == "iter-1":
        # First ensemble is 'iter-1', switch it so it is 'iter-0'
        ensemble_id1, ensemble_id2 = ensemble_id2, ensemble_id1

    resp: Response = dark_storage_client.get(f"/ensembles/{ensemble_id1}")
    ensemble_json = resp.json()
    assert ensemble_json["userdata"]["name"] == "iter-0", (
        f"\nexperiment_json: {json.dumps(experiment_json, indent=1)} \n\n"
        f"ensemble_json: {json.dumps(ensemble_json, indent=1)}"
    )

    resp: Response = dark_storage_client.get(f"/ensembles/{ensemble_id2}")
    ensemble_json2 = resp.json()

    assert ensemble_json2["userdata"]["name"] == "iter-1", (
        f"\nexperiment_json: {json.dumps(experiment_json, indent=1)} \n\n"
        f"ensemble_json2: {json.dumps(ensemble_json2, indent=1)}"
    )

    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id1}/responses/POLY_RES",
        params={"filter_on": json.dumps({"report_step": 0})},
        headers={"accept": "text/csv"},
    )
    stream = io.BytesIO(resp.content)
    record_df1 = pd.read_csv(stream, index_col=0, float_precision="round_trip")
    assert len(record_df1.columns) == 10
    assert len(record_df1.index) == 3

    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id1}/responses/POLY_RES",
        params={"filter_on": json.dumps({"report_step": 0})},
        headers={"accept": "application/x-parquet"},
    )
    stream = io.BytesIO(resp.content)
    record_df1 = pd.read_parquet(stream)
    assert len(record_df1.columns) == 10
    assert len(record_df1.index) == 3


@pytest.mark.integration_test
def test_get_summary_response(
    copy_snake_oil_case_storage, dark_storage_client_snake_oil
):
    resp: Response = dark_storage_client_snake_oil.get("/experiments")
    experiment_json = resp.json()

    assert len(experiment_json) == 1

    # ensemble_experiment, so only 1 ensemble
    assert len(experiment_json[0]["ensemble_ids"]) == 1

    ensemble_id = experiment_json[0]["ensemble_ids"][0]

    resp_ensemble: Response = dark_storage_client_snake_oil.get(
        f"/ensembles/{ensemble_id}"
    ).json()
    userdata = resp_ensemble["userdata"]

    assert resp_ensemble["size"] == 5
    # User data has entry started_at which changes with test every run.
    # Therefore, we match on all keys except that one.
    expected_userdata = {
        "name": "default_0",
        "experiment_name": "ensemble-experiment",
    }
    assert all(
        userdata[key] == expected_userdata[key]
        for key in userdata
        if key != "started_at"
    )

    resp_response: Response = dark_storage_client_snake_oil.get(
        f"/ensembles/{ensemble_id}/responses/FOPR",
        headers={"accept": "text/csv"},
    )
    stream = io.BytesIO(resp_response.content)
    record_df = pd.read_csv(stream, index_col=0, float_precision="round_trip")
    assert len(record_df.columns) == 200
    assert len(record_df.index) == 5


@pytest.mark.integration_test
def test_get_ensemble_parameters(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    experiment_json = resp.json()[0]

    assert experiment_json["parameters"] == {
        "a": [
            {
                "key": "COEFFS:a",
                "transformation": "UNIFORM",
                "dimensionality": 1,
                "userdata": {"data_origin": "GEN_KW"},
            }
        ],
        "b": [
            {
                "key": "COEFFS:b",
                "transformation": "UNIFORM",
                "dimensionality": 1,
                "userdata": {"data_origin": "GEN_KW"},
            }
        ],
        "c": [
            {
                "key": "COEFFS:c",
                "transformation": "UNIFORM",
                "dimensionality": 1,
                "userdata": {"data_origin": "GEN_KW"},
            }
        ],
    }


@pytest.mark.integration_test
def test_refresh_facade(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.post("/updates/facade")
    assert resp.status_code == 200


@pytest.mark.integration_test
def test_get_experiment_observations(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    experiment_json = resp.json()
    experiment_id = experiment_json[0]["id"]

    resp: Response = dark_storage_client.get(
        f"/experiments/{experiment_id}/observations"
    )
    response_json = resp.json()

    assert len(response_json) == 1
    assert response_json[0]["name"] == "POLY_OBS"
    assert len(response_json[0]["errors"]) == 5
    assert len(response_json[0]["values"]) == 5
    assert len(response_json[0]["x_axis"]) == 5


@pytest.mark.integration_test
def test_get_record_observations(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    answer_json = resp.json()
    ensemble_id = answer_json[0]["ensemble_ids"][0]

    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id}/responses/POLY_RES/observations",
    )
    response_json = resp.json()

    assert len(response_json) == 1
    assert response_json[0]["name"] == "POLY_OBS"
    assert len(response_json[0]["errors"]) == 5
    assert len(response_json[0]["values"]) == 5
    assert len(response_json[0]["x_axis"]) == 5


@pytest.mark.integration_test
def test_misfit_endpoint(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    experiment_json = resp.json()
    ensemble_id = experiment_json[0]["ensemble_ids"][0]

    resp: Response = dark_storage_client.get(
        "/compute/misfits",
        params={
            "filter_on": json.dumps({"report_step": 0}),
            "ensemble_id": ensemble_id,
            "response_name": "POLY_RES",
        },
        headers={"accept": "text/csv"},
    )
    stream = io.BytesIO(resp.content)
    misfit = pd.read_csv(stream, index_col=0, float_precision="round_trip")

    assert_array_equal(misfit.columns, ["0", "2", "4", "6", "8"])
    assert misfit.shape == (3, 5)


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "coeffs",
    [
        "COEFFS:a",
        "COEFFS:b",
        "COEFFS:c",
    ],
)
def test_get_coeffs_records(poly_example_tmp_dir, dark_storage_client, coeffs):
    resp: Response = dark_storage_client.get("/experiments")
    answer_json = resp.json()
    ensemble_id = answer_json[0]["ensemble_ids"][0]

    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id}/parameters/{coeffs}/",
        headers={"accept": "application/x-parquet"},
    )

    stream = io.BytesIO(resp.content)
    dataframe = pd.read_parquet(stream)

    assert all(dataframe.index.values == [1, 2, 4])
    assert dataframe.index.name == "Realization"
    assert dataframe.shape == (3, 1)
