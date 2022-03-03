import pytest
from performance_utils import dark_storage_app
from starlette.testclient import TestClient
import pandas as pd
import io
from requests import Response
import json


def get_single_record_csv(dark_storage_client, ensemble_id1, keyword, poly_ran):
    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id1}/records/{keyword}?realization_index={poly_ran['reals'] - 1}"
    )
    stream = io.BytesIO(resp.content)
    record_df1_indexed = pd.read_csv(stream, index_col=0, float_precision="round_trip")
    assert len(record_df1_indexed.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1_indexed.index) == 1


def get_observations(dark_storage_client, ensemble_id1, keyword, poly_ran):
    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id1}/records/{keyword}/observations"
    )
    stream = io.BytesIO(resp.content)
    json.load(stream)


def get_single_record_parquet(dark_storage_client, ensemble_id1, keyword, poly_ran):
    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id1}/records/{keyword}?realization_index={poly_ran['reals'] - 1}",
        headers={"accept": "application/x-parquet"},
    )
    stream = io.BytesIO(resp.content)
    record_df1_indexed = pd.read_parquet(stream)
    assert len(record_df1_indexed.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1_indexed.index) == 1


def get_record_parquet(dark_storage_client, ensemble_id1, keyword, poly_ran):
    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id1}/records/{keyword}",
        headers={"accept": "application/x-parquet"},
    )
    stream = io.BytesIO(resp.content)
    record_df1 = pd.read_parquet(stream)
    assert len(record_df1.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1.index) == poly_ran["reals"]


def get_record_csv(dark_storage_client, ensemble_id1, keyword, poly_ran):
    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id1}/records/{keyword}",
        headers={"accept": "text/csv"},
    )
    stream = io.BytesIO(resp.content)
    record_df1 = pd.read_csv(stream, index_col=0, float_precision="round_trip")
    assert len(record_df1.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1.index) == poly_ran["reals"]


def get_result(dark_storage_client, ensemble_id1, keyword, poly_ran):
    resp: Response = dark_storage_client.get(
        f"/ensembles/{ensemble_id1}/responses/{keyword}/data"
    )
    stream = io.BytesIO(resp.content)
    response_df1 = pd.read_csv(stream, index_col=0, float_precision="round_trip")

    assert len(response_df1.columns) == poly_ran["gen_data_entries"]
    assert len(response_df1.index) == poly_ran["reals"]


def get_parameters(dark_storage_client, ensemble_id1, keyword, poly_ran):
    resp: Response = dark_storage_client.get(f"/ensembles/{ensemble_id1}/parameters")
    parameters_json = resp.json()
    assert (
        len(parameters_json)
        == poly_ran["parameter_entries"] * poly_ran["parameter_count"]
    )


@pytest.mark.skip
@pytest.mark.parametrize(
    "function",
    [
        get_result,
        get_record_parquet,
        get_record_csv,
        get_single_record_parquet,
        get_observations,
        get_parameters,
    ],
)
@pytest.mark.parametrize(
    "keyword", ["summary", "gen_data", "summary_with_obs", "gen_data_with_obs"]
)
@pytest.mark.integration_test
def test_dark_storage_performance(benchmark, poly_ran, monkeypatch, function, keyword):

    key = {
        "summary": "PSUM1",
        "gen_data": "POLY_RES_1@0",
        "summary_with_obs": "PSUM0",
        "gen_data_with_obs": "POLY_RES_0@0",
    }[keyword]

    with poly_ran["folder"].as_cwd(), dark_storage_app(monkeypatch) as app:
        dark_storage_client = TestClient(app)
        resp: Response = dark_storage_client.get("/experiments")
        experiment_json = resp.json()

        ensemble_id1 = experiment_json[0]["ensemble_ids"][0]

        resp: Response = dark_storage_client.get(f"/ensembles/{ensemble_id1}")
        ensemble_json = resp.json()

        assert key in ensemble_json["response_names"]
        benchmark(function, dark_storage_client, ensemble_id1, key, poly_ran)
