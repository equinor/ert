import os

import pytest
from performance_utils import make_poly_template, dark_storage_app
from argparse import ArgumentParser
from ert_shared.cli import ENSEMBLE_EXPERIMENT_MODE
from ert_shared.cli.main import run_cli
from ert_shared.main import ert_parser
from starlette.testclient import TestClient
import pandas as pd
import io
from requests import Response
from pytest import fixture
import py
import json
from pathlib import Path

def make_case(reals, x_size):
    return {
        "gen_data_count": 2,
        "gen_data_entries": x_size,
        "summary_data_entries": x_size,
        "reals": reals,
        "summary_data_count": 2,
        "sum_obs_count": 1,
        "gen_obs_count": 1,
        "sum_obs_every": 2,
        "gen_obs_every": 2,
        "parameter_entries": 3,
        "parameter_count": 1,
        "ministeps": 1,
    }


cases_to_run = [
    make_case(reals=10, x_size=20),
    make_case(reals=10, x_size=200),
    make_case(reals=10, x_size=2000),
    make_case(reals=100, x_size=20),
    make_case(reals=100, x_size=200),
    make_case(reals=100, x_size=2000),
    make_case(reals=1000, x_size=20000),
]


@fixture(
    scope="session",
    params=[
        pytest.param(
            params,
            marks=(
                pytest.mark.slow
                if params["reals"] > 10
                or params["gen_data_entries"] > 20
                or params["summary_data_entries"] > 20
                else []
            ),
        )
        for params in cases_to_run
    ],
    ids=[
        f"gen_x: {params['gen_data_entries']}, "
        f"sum_x: {params['summary_data_entries']} "
        f"reals: {params['reals']}"
        for params in cases_to_run
    ],
)
def poly_ran(request, source_root, tmp_path_factory):
    tmpdir = py.path.local(tmp_path_factory.mktemp("my_poly_tmp"))
    params = request.param
    params.update()

    poly_folder = make_poly_template(tmpdir, source_root, **params)
    params["folder"] = poly_folder

    with poly_folder.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )
        run_cli(parsed)
        print(poly_folder)
        files = list(x for x in Path(".").rglob("*") if "storage" not in x.parts)
        files.sort()
        for file in files:
            print(f"{file}, {os.stat(file).st_size}")

    yield params

    # shutil.rmtree(poly_folder)


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
