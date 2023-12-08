import io
import json
import uuid

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from requests import Response


def test_get_experiment(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    answer_json = resp.json()
    assert len(answer_json) == 1
    assert "ensemble_ids" in answer_json[0]
    assert len(answer_json[0]["ensemble_ids"]) == 2
    assert "name" in answer_json[0]
    assert answer_json[0]["name"] == "default"


def test_get_ensemble(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    experiment_json = resp.json()
    assert len(experiment_json) == 1
    assert len(experiment_json[0]["ensemble_ids"]) == 2

    ensemble_id = experiment_json[0]["ensemble_ids"][0]

    resp: Response = dark_storage_client.get(f"/ensembles/{ensemble_id}")
    ensemble_json = resp.json()

    assert ensemble_json["experiment_id"] == experiment_json[0]["id"]
    assert ensemble_json["userdata"]["name"] in ("alpha", "beta")


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
    assert ensembles_json[0]["userdata"]["name"] in ("alpha", "beta")


def test_get_responses_with_observations(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    experiment_json = resp.json()
    ensemble_id = experiment_json[0]["ensemble_ids"][1]

    resp: Response = dark_storage_client.get(f"/ensembles/{ensemble_id}/responses")
    ensemble_json = resp.json()

    assert "POLY_RES@0" in ensemble_json
    assert "has_observations" in ensemble_json["POLY_RES@0"]
    assert ensemble_json["POLY_RES@0"]["has_observations"] is True


def test_get_response(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    experiment_json = resp.json()

    assert len(experiment_json[0]["ensemble_ids"]) == 2, experiment_json

    ensemble_id1 = experiment_json[0]["ensemble_ids"][0]
    ensemble_id2 = experiment_json[0]["ensemble_ids"][1]

    # Make sure the order is correct
    resp: Response = dark_storage_client.get(f"/ensembles/{ensemble_id1}")
    if resp.json()["userdata"]["name"] == "beta":
        # First ensemble is 'beta', switch it so it is 'alpha'
        ensemble_id1, ensemble_id2 = ensemble_id2, ensemble_id1

    resp: Response = dark_storage_client.get(f"/ensembles/{ensemble_id1}")
    ensemble_json = resp.json()
    assert ensemble_json["userdata"]["name"] == "alpha", (
        f"\nexperiment_json: {json.dumps(experiment_json, indent=1)} \n\n"
        f"ensemble_json: {json.dumps(ensemble_json, indent=1)}"
    )
    assert ensemble_json["response_names"][0] == "POLY_RES@0"

    resp: Response = dark_storage_client.get(f"/ensembles/{ensemble_id2}")
    ensemble_json2 = resp.json()

    assert ensemble_json2["userdata"]["name"] == "beta", (
        f"\nexperiment_json: {json.dumps(experiment_json, indent=1)} \n\n"
        f"ensemble_json2: {json.dumps(ensemble_json2, indent=1)}"
    )
    assert ensemble_json2["response_names"][0] == "POLY_RES@0"

    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id1}/responses/POLY_RES@0/data"
    )
    stream = io.BytesIO(resp.content)
    response_df1 = pd.read_csv(stream, index_col=0, float_precision="round_trip")

    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id2}/responses/POLY_RES@0/data"
    )
    stream = io.BytesIO(resp.content)
    response_df2 = pd.read_csv(stream, index_col=0, float_precision="round_trip")

    assert len(response_df1.columns) == 10, (
        f"\nexperiment_json: {json.dumps(experiment_json, indent=1)} \n\n"
        f"ensemble_json: {json.dumps(ensemble_json, indent=1)}\n"
        f" status_code: {resp.status_code}"
    )
    assert len(response_df1.index) == 3

    assert len(response_df2.columns) == 10
    assert len(response_df2.index) == 3

    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id1}/records/POLY_RES@0",
        headers={"accept": "text/csv"},
    )
    stream = io.BytesIO(resp.content)
    record_df1 = pd.read_csv(stream, index_col=0, float_precision="round_trip")
    assert len(record_df1.columns) == 10
    assert len(record_df1.index) == 3

    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id1}/records/POLY_RES@0",
        headers={"accept": "application/x-parquet"},
    )
    stream = io.BytesIO(resp.content)
    record_df1 = pd.read_parquet(stream)
    assert len(record_df1.columns) == 10
    assert len(record_df1.index) == 3

    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id1}/records/POLY_RES@0?realization_index=2"
    )
    stream = io.BytesIO(resp.content)
    record_df1_indexed = pd.read_csv(stream, index_col=0, float_precision="round_trip")
    assert len(record_df1_indexed.columns) == 10
    assert len(record_df1_indexed.index) == 1


def test_get_ensemble_parameters(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    answer_json = resp.json()
    ensemble_id = answer_json[0]["ensemble_ids"][0]

    resp: Response = dark_storage_client.get(f"/ensembles/{ensemble_id}/parameters")
    parameters_json = resp.json()

    assert len(parameters_json) == 3
    assert parameters_json[0] == {
        "labels": [],
        "name": "COEFFS:a",
        "userdata": {"data_origin": "GEN_KW"},
    }
    assert parameters_json[1] == {
        "labels": [],
        "name": "COEFFS:b",
        "userdata": {"data_origin": "GEN_KW"},
    }
    assert parameters_json[2] == {
        "labels": [],
        "name": "COEFFS:c",
        "userdata": {"data_origin": "GEN_KW"},
    }


def test_refresh_facade(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.post("/updates/facade")
    assert resp.status_code == 200


def test_get_experiment_observations(poly_example_tmp_dir, dark_storage_client):
    experiment_id = uuid.UUID(int=0)
    resp: Response = dark_storage_client.get(
        f"/experiments/{experiment_id}/observations"
    )
    response_json = resp.json()

    assert len(response_json) == 1
    assert response_json[0]["name"] == "POLY_OBS"
    assert len(response_json[0]["errors"]) == 5
    assert len(response_json[0]["values"]) == 5
    assert len(response_json[0]["x_axis"]) == 5


def test_get_record_observations(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    answer_json = resp.json()
    ensemble_id = answer_json[0]["ensemble_ids"][0]

    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id}/records/POLY_RES@0/observations"
    )
    response_json = resp.json()

    assert len(response_json) == 1
    assert response_json[0]["name"] == "POLY_OBS"
    assert len(response_json[0]["errors"]) == 5
    assert len(response_json[0]["values"]) == 5
    assert len(response_json[0]["x_axis"]) == 5


def test_misfit_endpoint(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    experiment_json = resp.json()
    ensemble_id = experiment_json[0]["ensemble_ids"][0]

    resp: Response = dark_storage_client.get(
        f"/compute/misfits?ensemble_id={ensemble_id}&response_name=POLY_RES@0"
    )
    stream = io.BytesIO(resp.content)
    misfit = pd.read_csv(stream, index_col=0, float_precision="round_trip")

    assert_array_equal(misfit.columns, ["0", "2", "4", "6", "8"])
    assert misfit.shape == (3, 5)


def test_get_record_labels(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    answer_json = resp.json()
    ensemble_id = answer_json[0]["ensemble_ids"][0]
    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id}/records/POLY_RES@0/labels"
    )
    labels = resp.json()

    assert labels == []


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
        f"/ensembles/{ensemble_id}/records/{coeffs}/",
        headers={"accept": "application/x-parquet"},
    )

    stream = io.BytesIO(resp.content)
    dataframe = pd.read_parquet(stream)

    assert all(dataframe.index.values == [1, 2, 4])
    assert dataframe.index.name == "Realization"
    assert dataframe.shape == tuple([3, 1])
