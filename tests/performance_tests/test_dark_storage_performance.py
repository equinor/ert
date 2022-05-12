import pytest

from ert_shared.async_utils import run_in_loop
import pandas as pd
import io
import os

from ert_shared.libres_facade import LibresFacade
from res.enkf import EnKFMain, ResConfig
from ert_shared.dark_storage.endpoints import experiments, ensembles, records, responses


def get_single_record_csv(ert, ensemble_id1, keyword, poly_ran):
    csv = run_in_loop(
        records.get_ensemble_record(
            res=ert,
            name=keyword,
            ensemble_id=ensemble_id1,
            realization_index=poly_ran["reals"] - 1,
        )
    ).body
    record_df1_indexed = pd.read_csv(
        io.BytesIO(csv), index_col=0, float_precision="round_trip"
    )
    assert len(record_df1_indexed.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1_indexed.index) == 1


def get_observations(ert, ensemble_id1, keyword: str, poly_ran):
    obs = run_in_loop(
        records.get_record_observations(res=ert, ensemble_id=ensemble_id1, name=keyword)
    )

    if "PSUM" in keyword:
        n = int(keyword[4:])
        if n < poly_ran["sum_obs_count"]:
            count = poly_ran["summary_data_entries"] // poly_ran["sum_obs_every"]
            assert len(obs) == 1
            assert len(obs[0].errors) == count
            assert len(obs[0].x_axis) == count
            assert len(obs[0].values) == count
        else:
            assert len(obs) == 0

    elif "POLY_RES_" in keyword:
        n = int(keyword.split("@")[0][9:])
        if n < poly_ran["gen_obs_count"]:
            count = poly_ran["gen_data_entries"] // poly_ran["gen_obs_every"]
            assert len(obs) == 1
            assert len(obs[0].errors) == count
            assert len(obs[0].x_axis) == count
            assert len(obs[0].values) == count
        else:
            assert len(obs) == 0
    else:
        assert False, f"should never get here, keyword is {keyword}"


def get_single_record_parquet(ert, ensemble_id1, keyword, poly_ran):
    parquet = run_in_loop(
        records.get_ensemble_record(
            res=ert,
            name=keyword,
            ensemble_id=ensemble_id1,
            realization_index=poly_ran["reals"] - 1,
            accept="application/x-parquet",
        )
    ).body
    record_df1_indexed = pd.read_parquet(io.BytesIO(parquet))
    assert len(record_df1_indexed.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1_indexed.index) == 1


def get_record_parquet(ert, ensemble_id1, keyword, poly_ran):
    parquet = run_in_loop(
        records.get_ensemble_record(
            res=ert,
            name=keyword,
            ensemble_id=ensemble_id1,
            accept="application/x-parquet",
        )
    ).body
    record_df1 = pd.read_parquet(io.BytesIO(parquet))
    assert len(record_df1.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1.index) == poly_ran["reals"]


def get_record_csv(ert, ensemble_id1, keyword, poly_ran):
    csv = run_in_loop(
        records.get_ensemble_record(res=ert, name=keyword, ensemble_id=ensemble_id1)
    ).body
    record_df1 = pd.read_csv(io.BytesIO(csv), index_col=0, float_precision="round_trip")
    assert len(record_df1.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1.index) == poly_ran["reals"]


def get_result(ert, ensemble_id1, keyword, poly_ran):
    csv = run_in_loop(
        responses.get_ensemble_response_dataframe(
            res=ert, ensemble_id=ensemble_id1, response_name=keyword
        )
    ).body
    response_df1 = pd.read_csv(
        io.BytesIO(csv), index_col=0, float_precision="round_trip"
    )
    assert len(response_df1.columns) == poly_ran["gen_data_entries"]
    assert len(response_df1.index) == poly_ran["reals"]


def get_parameters(ert, ensemble_id1, keyword, poly_ran):
    parameters_json = run_in_loop(
        records.get_ensemble_parameters(res=ert, ensemble_id=ensemble_id1)
    )
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
def test_direct_dark_performance(benchmark, poly_ran, monkeypatch, function, keyword):

    key = {
        "summary": "PSUM1",
        "gen_data": "POLY_RES_1@0",
        "summary_with_obs": "PSUM0",
        "gen_data_with_obs": "POLY_RES_0@0",
    }[keyword]

    with poly_ran["folder"].as_cwd():
        config = ResConfig("poly.ert")
        os.chdir(config.config_path)
        ert = EnKFMain(config, strict=True)
        enkf_facade = LibresFacade(ert)
        experiment_json = experiments.get_experiments(res=enkf_facade)
        ensemble_id1 = experiment_json[0].ensemble_ids[0]
        ensemble_json = ensembles.get_ensemble(
            res=enkf_facade, ensemble_id=ensemble_id1
        )
        assert key in ensemble_json.response_names
        benchmark(function, enkf_facade, ensemble_id1, key, poly_ran)
