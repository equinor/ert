import json
import os
from argparse import ArgumentParser

from requests import Response

from ert_shared.cli import ENSEMBLE_SMOOTHER_MODE
from ert_shared.cli.main import run_cli
from ert_shared.main import ert_parser


def test_my_get_experiment(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.post(
        "/gql", json={"query": "{experiments{name}}"}
    )
    answer_json = resp.json()
    print(answer_json)
    assert "experiments" in answer_json["data"]
    assert len(answer_json["data"]["experiments"]) == 1
    assert "name" in answer_json["data"]["experiments"][0]
    assert answer_json["data"]["experiments"][0]["name"] == "default"


def test_my_get_enesembles(poly_example_tmp_dir, dark_storage_client):
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--target-case",
            "poly_runpath_file",
            "--realizations",
            "1,2,4",
            "poly.ert",
        ],
    )

    run_cli(parsed)

    resp: Response = dark_storage_client.post(
        "/gql", json={"query": "{experiments{ensembles{userdata}}}"}
    )
    answer_json = resp.json()
    assert "experiments" in answer_json["data"]
    assert len(answer_json["data"]["experiments"]) == 1

    assert "ensembles" in answer_json["data"]["experiments"][0]
    assert len(answer_json["data"]["experiments"][0]["ensembles"]) == 2

    assert "userdata" in answer_json["data"]["experiments"][0]["ensembles"][0]
    userdata = json.loads(
        answer_json["data"]["experiments"][0]["ensembles"][0]["userdata"]
    )
    assert "name" in userdata
    assert userdata["name"] == "default"

    assert "userdata" in answer_json["data"]["experiments"][0]["ensembles"][1]
    userdata = json.loads(
        answer_json["data"]["experiments"][0]["ensembles"][1]["userdata"]
    )
    assert "name" in userdata
    assert userdata["name"] == "poly_runpath_file"
