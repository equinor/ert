import io

import pandas as pd
import pytest

from ert.async_utils import run_in_loop
from ert.config import ErtConfig
from ert.dark_storage.endpoints import ensembles, experiments, records, responses
from ert.enkf_main import EnKFMain
from ert.libres_facade import LibresFacade
from ert.storage import open_storage


def get_single_record_csv(ert, storage, ensemble_id1, keyword, poly_ran):
    csv = run_in_loop(
        records.get_ensemble_record(
            res=ert,
            db=storage,
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


def get_observations(ert, storage, ensemble_id1, keyword: str, poly_ran):
    obs = run_in_loop(
        records.get_record_observations(
            res=ert, db=storage, ensemble_id=ensemble_id1, name=keyword
        )
    )

    if "PSUM" in keyword:
        n = int(keyword[4:])
        if n < poly_ran["sum_obs_count"]:
            count = poly_ran["summary_data_entries"] // poly_ran["sum_obs_every"]
            assert len(obs) == 1
            assert obs[0].errors[0] == 0.1
            assert obs[0].x_axis[0] == "2010-01-02T00:00:00.000000000"
            assert obs[0].values[0] == 2.6357
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
        raise AssertionError(f"should never get here, keyword is {keyword}")


def get_single_record_parquet(ert, storage, ensemble_id1, keyword, poly_ran):
    parquet = run_in_loop(
        records.get_ensemble_record(
            res=ert,
            db=storage,
            name=keyword,
            ensemble_id=ensemble_id1,
            realization_index=poly_ran["reals"] - 1,
            accept="application/x-parquet",
        )
    ).body
    record_df1_indexed = pd.read_parquet(io.BytesIO(parquet))
    assert len(record_df1_indexed.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1_indexed.index) == 1


def get_record_parquet(ert, storage, ensemble_id1, keyword, poly_ran):
    parquet = run_in_loop(
        records.get_ensemble_record(
            res=ert,
            db=storage,
            name=keyword,
            ensemble_id=ensemble_id1,
            accept="application/x-parquet",
        )
    ).body
    record_df1 = pd.read_parquet(io.BytesIO(parquet))
    assert len(record_df1.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1.index) == poly_ran["reals"]


def get_record_csv(ert, storage, ensemble_id1, keyword, poly_ran):
    csv = run_in_loop(
        records.get_ensemble_record(
            res=ert, db=storage, name=keyword, ensemble_id=ensemble_id1
        )
    ).body
    record_df1 = pd.read_csv(io.BytesIO(csv), index_col=0, float_precision="round_trip")
    assert len(record_df1.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1.index) == poly_ran["reals"]


def get_result(ert, storage, ensemble_id1, keyword, poly_ran):
    csv = run_in_loop(
        responses.get_ensemble_response_dataframe(
            res=ert, db=storage, ensemble_id=ensemble_id1, response_name=keyword
        )
    ).body
    response_df1 = pd.read_csv(
        io.BytesIO(csv), index_col=0, float_precision="round_trip"
    )
    assert len(response_df1.columns) == poly_ran["gen_data_entries"]
    assert len(response_df1.index) == poly_ran["reals"]


def get_parameters(ert, storage, ensemble_id1, keyword, poly_ran):
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
def test_direct_dark_performance(
    benchmark, template_config, monkeypatch, function, keyword
):
    key = {
        "summary": "PSUM1",
        "gen_data": "POLY_RES_1@0",
        "summary_with_obs": "PSUM0",
        "gen_data_with_obs": "POLY_RES_0@0",
    }[keyword]

    with template_config["folder"].as_cwd():
        config = ErtConfig.from_file("poly.ert")
        ert = EnKFMain(config)
        enkf_facade = LibresFacade(ert)
        storage = open_storage(enkf_facade.enspath)
        experiment_json = experiments.get_experiments(res=enkf_facade, db=storage)
        ensemble_json_default = None
        ensemble_id_default = None
        for ensemble_id in experiment_json[0].ensemble_ids:
            ensemble_json = ensembles.get_ensemble(
                res=enkf_facade, db=storage, ensemble_id=ensemble_id
            )
            if ensemble_json.userdata["name"] == "default":
                ensemble_json_default = ensemble_json
                ensemble_id_default = ensemble_id
        assert key in ensemble_json_default.response_names
        benchmark(
            function, enkf_facade, storage, ensemble_id_default, key, template_config
        )
