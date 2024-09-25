import logging
from datetime import datetime, timedelta
from textwrap import dedent

import pytest
from pandas.core.frame import DataFrame
from resdata.summary import Summary

from ert.config import ErtConfig
from ert.enkf_main import sample_prior
from ert.libres_facade import LibresFacade
from ert.storage import open_storage


@pytest.fixture
def facade(snake_oil_case):
    return LibresFacade(snake_oil_case)


@pytest.fixture
def storage(facade):
    with open_storage(facade.enspath, mode="w") as storage:
        yield storage


@pytest.fixture
def empty_case(facade, storage):
    experiment_id = storage.create_experiment()
    return storage.create_ensemble(
        experiment_id, name="new_case", ensemble_size=facade.get_ensemble_size()
    )


def test_keyword_type_checks(snake_oil_default_storage):
    assert "BPR:1,3,8" in snake_oil_default_storage.get_summary_keyset()


def test_keyword_type_checks_missing_key(snake_oil_default_storage):
    assert "nokey" not in snake_oil_default_storage.get_summary_keyset()


@pytest.mark.filterwarnings("ignore:.*Use load_responses.*:DeprecationWarning")
def test_data_fetching_missing_key(empty_case):
    data = [
        empty_case.load_all_summary_data(["nokey"]),
        empty_case.load_all_gen_kw_data("nokey", None),
    ]

    for dataframe in data:
        assert isinstance(dataframe, DataFrame)
        assert dataframe.empty


@pytest.mark.filterwarnings("ignore:.*Use load_responses.*:DeprecationWarning")
def test_summary_data_verify_indices_and_values(
    caplog, snake_oil_default_storage, snapshot
):
    with caplog.at_level(logging.WARNING):
        data = snake_oil_default_storage.load_all_summary_data(["FOPR"])
        data = data.unstack(level="Realization")
        snapshot.assert_match(
            data.iloc[:5].to_csv(),
            "summary_head.csv",
        )
        snapshot.assert_match(
            data.iloc[-5:].to_csv(),
            "summary_tail.csv",
        )

        assert data.shape == (200, 5)
        assert "contains duplicate timestamps" not in caplog.text


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_kw_log_appended_extra():
    with open("config_file.ert", "w", encoding="utf-8") as fout:
        fout.write(
            dedent(
                """
        NUM_REALIZATIONS 1
        GEN_KW KW_NAME template.txt kw.txt prior.txt
        """
            )
        )
    with open("template.txt", "w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD <MY_KEYWORD>")
    with open("prior.txt", "w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD LOGNORMAL 1 2")


@pytest.mark.filterwarnings("ignore:.*Use load_responses.*:DeprecationWarning")
def test_summary_collector(
    monkeypatch, snake_oil_case_storage, snake_oil_default_storage, snapshot
):
    monkeypatch.setenv("TZ", "CET")  # The ert_statoil case was generated in CET

    data = snake_oil_default_storage.load_all_summary_data()
    data = data.drop(["TIME"], axis=1)
    snapshot.assert_match(
        data.iloc[:4].round(4).to_csv(),
        "summary_collector_1.csv",
    )
    assert data.shape == (1000, 44)
    with pytest.raises(KeyError):
        # realization 60:
        _ = data.loc[60]

    data = snake_oil_default_storage.load_all_summary_data(["WWCT:OP1", "WWCT:OP2"])
    snapshot.assert_match(data.iloc[:4].to_csv(), "summary_collector_2.csv")
    assert data.shape == (1000, 2)
    with pytest.raises(KeyError):
        _ = data["FOPR"]

    realization_index = 4
    data = snake_oil_default_storage.load_all_summary_data(
        ["WWCT:OP1", "WWCT:OP2"],
        realization_index=realization_index,
    )
    snapshot.assert_match(data.iloc[:4].to_csv(), "summary_collector_3.csv")
    assert data.shape == (200, 2)
    non_existing_realization_index = 150
    with pytest.raises(IndexError):
        _ = snake_oil_default_storage.load_all_summary_data(
            ["WWCT:OP1", "WWCT:OP2"],
            realization_index=non_existing_realization_index,
        )


def test_misfit_collector(snake_oil_case_storage, snake_oil_default_storage, snapshot):
    facade = LibresFacade(snake_oil_case_storage)
    data = facade.load_all_misfit_data(snake_oil_default_storage)
    snapshot.assert_match(data.to_csv(), "misfit_collector.csv")

    with pytest.raises(KeyError):
        # realization 60:
        _ = data.loc[60]


def test_gen_kw_collector(snake_oil_default_storage, snapshot):
    data = snake_oil_default_storage.load_all_gen_kw_data()
    snapshot.assert_match(data.round(6).to_csv(), "gen_kw_collector.csv")

    with pytest.raises(KeyError):
        # realization 60:
        _ = data.loc[60]

    data = snake_oil_default_storage.load_all_gen_kw_data(
        "SNAKE_OIL_PARAM",
    )[["SNAKE_OIL_PARAM:OP1_PERSISTENCE", "SNAKE_OIL_PARAM:OP1_OFFSET"]]
    snapshot.assert_match(data.round(6).to_csv(), "gen_kw_collector_2.csv")

    with pytest.raises(KeyError):
        _ = data["SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE"]

    realization_index = 3
    data = snake_oil_default_storage.load_all_gen_kw_data(
        "SNAKE_OIL_PARAM",
        realization_index=realization_index,
    )["SNAKE_OIL_PARAM:OP1_PERSISTENCE"]
    snapshot.assert_match(data.round(6).to_csv(), "gen_kw_collector_3.csv")

    non_existing_realization_index = 150
    with pytest.raises((IndexError, KeyError)):
        _ = snake_oil_default_storage.load_all_gen_kw_data(
            "SNAKE_OIL_PARAM",
            realization_index=non_existing_realization_index,
        )["SNAKE_OIL_PARAM:OP1_PERSISTENCE"]


def test_get_observations(tmpdir):
    date = datetime(2014, 9, 10)
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 2

        ECLBASE ECLIPSE_CASE
        REFCASE ECLIPSE_CASE
        OBS_CONFIG observations
        """
        )
        observations = dedent(
            f"""
        SUMMARY_OBSERVATION FOPR_1
        {{
        VALUE   = 0.1;
        ERROR   = 0.05;
        DATE    = {(date + timedelta(days=1)).isoformat()};
        KEY     = FOPR;
        }};
        """
        )

        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fh:
            fh.writelines(observations)

        summary = Summary.writer("ECLIPSE_CASE", date, 3, 3, 3)
        summary.add_variable("FOPR", unit="SM3/DAY")
        t_step = summary.add_t_step(1, sim_days=1)
        t_step["FOPR"] = 1
        summary.fwrite()

        facade = LibresFacade.from_config_file("config.ert")
        assert "FOPR_1" in facade.get_observations()


def test_load_gen_kw_not_sorted(storage, tmpdir, snapshot):
    """
    This test checks two things, loading multiple parameters and
    loading log parameters.
    """
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 10
        GEN_KW PARAM_2 template.txt kw.txt prior.txt
        GEN_KW PARAM_1 template.txt kw.txt prior.txt
        RANDOM_SEED 1234
        """
        )
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD LOGUNIF 0.1 1")

        ert_config = ErtConfig.from_file("config.ert")

        experiment_id = storage.create_experiment(
            parameters=ert_config.ensemble_config.parameter_configuration
        )
        ensemble_size = 10
        ensemble = storage.create_ensemble(
            experiment_id, name="default", ensemble_size=ensemble_size
        )

        sample_prior(ensemble, range(ensemble_size), random_seed=1234)

        data = ensemble.load_all_gen_kw_data()
        snapshot.assert_match(data.round(12).to_csv(), "gen_kw_unsorted")
