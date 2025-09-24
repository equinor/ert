import os

import numpy as np
import polars as pl
import pytest

from ert.data import MeasuredData
from ert.data._measured_data import ResponseError
from ert.libres_facade import LibresFacade
from ert.storage import open_storage


@pytest.fixture()
def facade_snake_oil(snake_oil_case_storage):
    yield LibresFacade(snake_oil_case_storage)


@pytest.fixture
def create_measured_data(snake_oil_case_storage, snake_oil_default_storage):
    def func(*args, **kwargs):
        return MeasuredData(
            snake_oil_default_storage,
            *args,
            **kwargs,
        )

    return func


def test_history_obs(create_measured_data):
    fopr = create_measured_data(["FOPR"])
    fopr.remove_inactive_observations()

    assert fopr.data.shape == (7, 200)


@pytest.mark.integration_test
def test_summary_obs(create_measured_data):
    summary_obs = create_measured_data(["WOPR_OP1_72"])
    summary_obs.remove_inactive_observations()
    # Only one observation, we check the key_index is what we expect:
    assert (
        summary_obs.data.columns.get_level_values("key_index").values[0]
        == "2011-12-21 00:00:00.000"
    )


def test_gen_obs(create_measured_data):
    df = create_measured_data(["WPR_DIFF_1"])
    df.remove_inactive_observations()

    assert all(
        df.data.columns.get_level_values("key_index").values
        == ["199, 400", "199, 800", "199, 1200", "199, 1800"]
    )


def test_gen_obs_and_summary(create_measured_data):
    df = create_measured_data(["WPR_DIFF_1", "WOPR_OP1_9"])
    df.remove_inactive_observations()

    assert df.data.columns.get_level_values(0).to_list() == sorted(
        [
            "WPR_DIFF_1",
            "WPR_DIFF_1",
            "WPR_DIFF_1",
            "WPR_DIFF_1",
            "WOPR_OP1_9",
        ]
    )


@pytest.mark.parametrize(
    "obs_key, expected_msg",
    [
        ("FOPR", r"Observations: FOPR not in experiment"),
        ("WPR_DIFF_1", "Observations: WPR_DIFF_1 not in experiment"),
    ],
)
def test_no_storage(obs_key, expected_msg, storage):
    ensemble = storage.create_experiment().create_ensemble(
        name="empty", ensemble_size=10
    )

    with pytest.raises(
        KeyError,
        match=expected_msg,
    ):
        MeasuredData(ensemble, [obs_key])


def create_summary_observation():
    observations = ""
    rng = np.random.default_rng()
    values = rng.uniform(0, 1.5, 200)
    errors = values * 0.1
    for restart, (value, error) in enumerate(zip(values, errors, strict=False)):
        observations += f"""
    \nSUMMARY_OBSERVATION FOPR_{restart + 1}
{{
    VALUE   = {value};
    ERROR   = {error};
    RESTART = {restart + 1};
    KEY     = FOPR;
}};
    """
    return observations


def create_general_observation():
    observations = ""
    index_list = np.array(range(2000))
    index_list = [index_list[i : i + 4] for i in range(0, len(index_list), 4)]
    for nr, (i1, i2, i3, i4) in enumerate(index_list):
        observations += f"""
    \nGENERAL_OBSERVATION CUSTOM_DIFF_{nr}
{{
   DATA       = SNAKE_OIL_WPR_DIFF;
   INDEX_LIST = {i1},{i2},{i3},{i4};
   RESTART    = 199;
   OBS_FILE   = wpr_diff_obs.txt;
}};
    """
    return observations


def test_all_measured_snapshot(snapshot, snake_oil_storage, create_measured_data):
    """
    While there is no guarantee that this snapshot is 100% correct, it does represent
    the current state of loading from storage for the snake_oil case.
    """
    experiment = next(snake_oil_storage.experiments)
    obs_keys = experiment.observation_keys
    measured_data = create_measured_data(obs_keys)
    snapshot.assert_match(
        measured_data.data.round(10).to_csv(), "snake_oil_measured_output.csv"
    )


def test_that_measured_data_gives_error_on_missing_response(snake_oil_case_storage):
    with open_storage(snake_oil_case_storage.ens_path, mode="w") as storage:
        experiment = storage.get_experiment_by_name("ensemble-experiment")
        ensemble = experiment.get_ensemble_by_name("default_0")

        for real in range(ensemble.ensemble_size):
            # .save_responses() does not allow for saving directly with an empty ds
            ds_path = ensemble._realization_dir(real) / "summary.parquet"
            smry_df = pl.read_parquet(ds_path)
            os.remove(ds_path)
            smry_df.clear().write_parquet(ds_path)

        with pytest.raises(
            ResponseError, match="No response loaded for observation type: summary"
        ):
            MeasuredData(ensemble, ["FOPR"])
