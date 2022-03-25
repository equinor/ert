import json
import uuid
from argparse import ArgumentParser

from requests import Response

from ert_shared.cli import ENSEMBLE_SMOOTHER_MODE
from ert_shared.cli.main import run_cli
from ert_shared.main import ert_parser


def test_get_experiment(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.post(
        "/gql", json={"query": "{experiments{name, priors}}"}
    )
    answer_json = resp.json()
    print(answer_json)

    assert len(answer_json["data"]["experiments"]) == 1
    exp = answer_json["data"]["experiments"][0]
    assert exp["name"] == "default"
    priors = json.loads(exp["priors"])
    assert "COEFFS" in priors
    assert "COEFF_A" == priors["COEFFS"][0]["key"]
    assert "COEFF_B" == priors["COEFFS"][1]["key"]
    assert "COEFF_C" == priors["COEFFS"][2]["key"]


def test_get_ensembles(poly_example_tmp_dir, dark_storage_client):

    resp: Response = dark_storage_client.post(
        "/gql", json={"query": "{experiments{ensembles{userdata}}}"}
    )
    answer_json = resp.json()
    assert "experiments" in answer_json["data"]
    assert len(answer_json["data"]["experiments"]) == 1

    assert "ensembles" in answer_json["data"]["experiments"][0]
    assert len(answer_json["data"]["experiments"][0]["ensembles"]) == 3

    assert "userdata" in answer_json["data"]["experiments"][0]["ensembles"][0]
    userdata = json.loads(
        answer_json["data"]["experiments"][0]["ensembles"][0]["userdata"]
    )
    assert "name" in userdata
    assert userdata["name"] == "alpha"

    assert "userdata" in answer_json["data"]["experiments"][0]["ensembles"][1]
    userdata = json.loads(
        answer_json["data"]["experiments"][0]["ensembles"][1]["userdata"]
    )
    assert "name" in userdata
    assert userdata["name"] == "beta"


def test_get_response_names(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.post(
        "/gql", json={"query": "{experiments{ensembles{responseNames}}}"}
    )
    answer_json = resp.json()
    print(answer_json)
    response_names = answer_json["data"]["experiments"][0]["ensembles"][0][
        "responseNames"
    ]
    assert len(response_names) == 1
    assert "POLY_RES@0" in response_names


def test_get_responses(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.post(
        "/gql",
        json={
            "query": "{experiments{ensembles{responses{"
            "id, name, realizationIndex, timeCreated, timeUpdated, userdata}}}}"
        },
    )
    answer_json = resp.json()
    print(answer_json)
    responses = answer_json["data"]["experiments"][0]["ensembles"][0]["responses"]
    assert len(responses) == 3
    for real_idx, response in zip([1, 2, 4], responses):
        assert response["name"] == "POLY_RES@0"
        assert response["realizationIndex"] == real_idx


def test_query_ensemble_parameters(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.post(
        "/gql", json={"query": "{experiments{ensembles{id}}}"}
    )
    answer_json = resp.json()
    ensemble_id = answer_json["data"]["experiments"][0]["ensembles"][0]["id"]

    resp: Response = dark_storage_client.post(
        "/gql",
        json={"query": f'{{ensemble(id: "{ensemble_id}") {{parameters {{name}} }} }}'},
    )

    answer_json = resp.json()
    assert "ensemble" in answer_json["data"]
    assert len(answer_json["data"]["ensemble"]) == 1
    assert "parameters" in answer_json["data"]["ensemble"]
    assert len(answer_json["data"]["ensemble"]["parameters"]) == 3
