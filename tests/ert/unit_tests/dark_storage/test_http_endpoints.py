import datetime
import io
import json
import re

import pandas as pd
import pytest
from requests import Response
from starlette.testclient import TestClient

from ert.analysis.event import AnalysisCompleteEvent, AnalysisMatrixEvent, DataSection
from ert.config._observations import SummaryObservation
from ert.config._shapes import CircleShapeConfig, ShapeRegistry
from ert.config.rft_config import RFTConfig
from ert.dark_storage.common import get_storage_api_version
from ert.dark_storage.endpoints.observations import _get_observations
from ert.storage import open_storage
from tests.ert.defaults_generator import (
    create_breakthrough_observation,
    create_general_observation,
    create_rft_observation,
    create_seismic_observation,
    create_summary_observation,
)


@pytest.mark.slow
def test_get_experiment(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    answer_json = resp.json()
    assert len(answer_json) == 1
    assert "ensemble_ids" in answer_json[0]
    assert len(answer_json[0]["ensemble_ids"]) == 2
    assert "name" in answer_json[0]


@pytest.mark.slow
def test_get_storage_api_version(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/version")
    answer_json = resp.json()

    assert answer_json == get_storage_api_version()

    version_pattern = r"^\d+\.\d+$"

    assert re.match(version_pattern, answer_json), (
        f"Version format is invalid: {answer_json}"
    )


@pytest.mark.slow
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


@pytest.mark.slow
def test_get_responses_with_observations(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    experiment_json = resp.json()[0]

    assert experiment_json["observations"] == {"gen_data": {"POLY_RES": ["POLY_OBS"]}}
    assert experiment_json["responses"] == {
        "gen_data": {
            "type": "gen_data",
            "keys": ["POLY_RES"],
            "has_finalized_keys": True,
            "input_files": [
                "poly.out",
            ],
            "report_steps_list": [
                None,
            ],
        }
    }


@pytest.mark.slow
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


@pytest.mark.slow
def test_get_summary_response(
    copy_snake_oil_case_storage, dark_storage_client_snake_oil
):
    resp: Response = dark_storage_client_snake_oil.get("/experiments")
    experiments_json = resp.json()

    experiment_json = next(
        e for e in experiments_json if e.get("name") == "ensemble-experiment"
    )

    # ensemble_experiment, so only 1 ensemble
    assert len(experiment_json["ensemble_ids"]) == 1

    ensemble_id = experiment_json["ensemble_ids"][0]

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
        "has_func_eval": False,
        "has_gradient": False,
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


@pytest.mark.slow
def test_get_ensemble_parameters(poly_example_tmp_dir, dark_storage_client):
    resp: Response = dark_storage_client.get("/experiments")
    experiment_json = resp.json()[0]

    assert experiment_json["parameters"] == {
        "a": {
            "dimensionality": 1,
            "distribution": {
                "max": 1.0,
                "min": 0.0,
                "name": "uniform",
            },
            "forward_init": False,
            "group": "COEFFS",
            "input_source": "sampled",
            "name": "a",
            "type": "gen_kw",
            "update_strategy": "global",
        },
        "b": {
            "dimensionality": 1,
            "distribution": {
                "max": 2.0,
                "min": 0.0,
                "name": "uniform",
            },
            "forward_init": False,
            "group": "COEFFS",
            "input_source": "sampled",
            "name": "b",
            "type": "gen_kw",
            "update_strategy": "global",
        },
        "c": {
            "dimensionality": 1,
            "distribution": {
                "max": 5.0,
                "min": 0.0,
                "name": "uniform",
            },
            "forward_init": False,
            "group": "COEFFS",
            "input_source": "sampled",
            "name": "c",
            "type": "gen_kw",
            "update_strategy": "global",
        },
    }


@pytest.mark.slow
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
    assert response_json[0]["east"] == 5 * [None]
    assert response_json[0]["north"] == 5 * [None]
    assert response_json[0]["radius"] == 5 * [None]


def test_that_experiment_observations_endpoint_returns_localization(
    tmp_path, monkeypatch, dark_storage_app
):
    storage_path = tmp_path / "storage"
    with open_storage(storage_path, mode="w") as storage:
        shape_registry = ShapeRegistry()
        first_shape_id = shape_registry.register(
            CircleShapeConfig(east=5.5, north=7.5, radius=15.0)
        )
        second_shape_id = shape_registry.register(
            CircleShapeConfig(east=9.5, north=3.5, radius=20.0)
        )
        experiment = storage.create_experiment(
            name="test-experiment",
            experiment_config={
                "observations": [
                    SummaryObservation(
                        name="OBSERVATION_1",
                        value=1.23,
                        error=0.1,
                        key="FOPR",
                        date="2010-05-13",
                        shape_id=first_shape_id,
                    ).model_dump(mode="json"),
                    SummaryObservation(
                        name="OBSERVATION_2",
                        value=2.34,
                        error=0.2,
                        key="FGPR",
                        date="2010-05-14",
                        shape_id=second_shape_id,
                    ).model_dump(mode="json"),
                ],
                "shape_registry": shape_registry.model_dump(mode="json"),
            },
        )
        experiment.create_ensemble(name="prior", ensemble_size=1)

    monkeypatch.setenv("ERT_STORAGE_ENS_PATH", str(storage_path))
    with TestClient(dark_storage_app) as client:
        resp: Response = client.get(f"/experiments/{experiment.id}/observations")

    response_json = resp.json()
    assert len(response_json) == 2

    observations_by_name = {obs["name"]: obs for obs in response_json}

    assert observations_by_name["OBSERVATION_1"]["east"] == [5.5]
    assert observations_by_name["OBSERVATION_1"]["north"] == [7.5]
    assert observations_by_name["OBSERVATION_1"]["radius"] == [15.0]

    assert observations_by_name["OBSERVATION_2"]["east"] == [9.5]
    assert observations_by_name["OBSERVATION_2"]["north"] == [3.5]
    assert observations_by_name["OBSERVATION_2"]["radius"] == [20.0]


def test_blob_endpoint_includes_matrix_parameter_group_sizes(
    tmp_path, monkeypatch, dark_storage_app
):
    storage_path = tmp_path / "storage"

    with open_storage(storage_path, mode="w") as storage:
        experiment = storage.create_experiment(name="test-experiment")
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="test-ensemble"
        )
        ensemble.save_blob(
            AnalysisMatrixEvent(
                name="K",
                sparse=False,
                shape=(2, 2),
                data_type="float64",
                update_algorithm="enif",
                parameter_group_sizes={"PORO": 8, "PERM": 3},
                matrix_bytes=b"matrix",
            )
        )
        ensemble_id = ensemble.id

    monkeypatch.setenv("ERT_STORAGE_ENS_PATH", str(storage_path))

    with TestClient(dark_storage_app) as client:
        resp = client.get(f"/ensembles/{ensemble_id}/blobs")

    assert resp.status_code == 200

    [blob] = resp.json()
    assert blob["name"] == "K"
    assert blob["blob_info"]["parameter_group_sizes"] == {
        "PORO": 8,
        "PERM": 3,
    }


def test_that_blob_endpoint_returns_blob_bytes(tmp_path, monkeypatch, dark_storage_app):
    storage_path = tmp_path / "storage"
    with open_storage(storage_path, mode="w") as storage:
        experiment = storage.create_experiment(name="test-experiment")
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="test-ensemble"
        )
        ensemble.save_blob(
            AnalysisCompleteEvent(
                data=DataSection(
                    header=["observation_key", "status", "value"],
                    data=[("OBS", "Active", 1.5)],
                ),
                update_algorithm="ensemble_smoother",
            )
        )
        blob = ensemble.load_blobs()[0]
        blob_bytes = ensemble.load_blob(blob.uri)
        ensemble_id = ensemble.id

    monkeypatch.setenv("ERT_STORAGE_ENS_PATH", str(storage_path))
    with TestClient(dark_storage_app) as client:
        resp: Response = client.get(f"/ensembles/{ensemble_id}/blobs/{blob.uri}")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.content == blob_bytes


@pytest.mark.slow
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
    assert response_json[0]["east"] == 5 * [None]
    assert response_json[0]["north"] == 5 * [None]


@pytest.mark.slow
@pytest.mark.parametrize(
    "coeffs",
    [
        "a",
        "b",
        "c",
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

    assert all(dataframe.index.to_numpy() == [1, 2, 4])
    assert dataframe.index.name == "Realization"
    assert dataframe.shape == (3, 1)


def test_that_observations_are_sorted_on_x_axis_column(tmp_path):
    rft_config = RFTConfig(input_files=["DUMMY"])
    storage_path = tmp_path / "storage"
    observations = [
        *[
            create_summary_observation(name="SUMMARY_OBSERVATION", date=date)
            for date in ["2010-11-01", "2010-07-01", "2010-12-01"]
        ],
        *[
            create_general_observation(name="GENERAL_OBSERVATION", index=index)
            for index in [11, 7, 12]
        ],
        *[
            create_breakthrough_observation(
                name="BREAKTHROUGH_OBSERVATION",
                date=datetime.datetime(2010, month, 1),  # noqa: DTZ001
            )
            for month in [11, 7, 12]
        ],
        *[
            create_rft_observation(name="RFT_OBSERVATION", tvd=tvd)
            for tvd in [11.0, 7.0, 12.0]
        ],
        *[
            create_seismic_observation(
                name="SEISMIC_OBSERVATION",
                east=east,
                north=north,
            )
            for east, north in [
                (15.0, 25.0),
                (15.0 + 3, 25.0 + 4),
                (15.0 + 3 + 5, 25.0 + 4 + 12),
            ]
        ],
    ]

    with open_storage(storage_path, mode="w") as storage:
        experiment = storage.create_experiment(
            name="test-experiment",
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [obs.model_dump(mode="json") for obs in observations],
            },
        )
        experiment.create_ensemble(name="prior", ensemble_size=1)

    obs_with_x_axis = _get_observations(experiment)
    for observation in obs_with_x_axis:
        match observation["name"]:
            case "SUMMARY_OBSERVATION" | "BREAKTHROUGH_OBSERVATION":
                assert observation["x_axis"] == [
                    "2010-07-01T00:00:00.000",
                    "2010-11-01T00:00:00.000",
                    "2010-12-01T00:00:00.000",
                ]
            case "GENERAL_OBSERVATION":
                assert observation["x_axis"] == ["7", "11", "12"]
            case "RFT_OBSERVATION":
                assert observation["x_axis"] == ["7.0", "11.0", "12.0"]
            case "SEISMIC_OBSERVATION":
                assert observation["x_axis"] == ["0.0", "5.0", "18.0"]
