import os

import numpy as np
import polars as pl
import pytest

from ert.storage import open_storage
from ert.storage.local_ensemble import ResponseError


@pytest.fixture
def create_measured_data(snake_oil_case_storage, snake_oil_default_storage):
    def func(keys):
        return snake_oil_default_storage._load_measured_data(keys)

    return func


@pytest.mark.slow
def test_summary_obs(create_measured_data):
    summary_obs = create_measured_data(["WOPR_OP1_72"])
    # Only one observation, we check the key_index is what we expect:
    assert summary_obs["key_index"][0] == "2011-12-21 00:00:00.000"


@pytest.mark.slow
def test_gen_obs(create_measured_data):
    df = create_measured_data(["WPR_DIFF_1"])

    assert df["key_index"].to_list() == [
        "199, 400",
        "199, 800",
        "199, 1200",
        "199, 1800",
    ]


@pytest.mark.slow
def test_gen_obs_and_summary(create_measured_data):
    df = create_measured_data(["WPR_DIFF_1", "WOPR_OP1_9"])

    assert df.to_series(0).to_list() == sorted(
        [
            "WPR_DIFF_1",
            "WPR_DIFF_1",
            "WPR_DIFF_1",
            "WPR_DIFF_1",
            "WOPR_OP1_9",
        ]
    )


@pytest.mark.parametrize(
    ("obs_key", "expected_msg"),
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
        ensemble._load_measured_data([obs_key])


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


@pytest.mark.slow
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
            ensemble._load_measured_data(["FOPR"])


@pytest.mark.slow
def test_that_check_expected_keys_for_measured_data_raises_when_obs_and_std_are_missing(
    snake_oil_case_storage, snake_oil_default_storage
):
    frame_without_obs_and_std = pl.DataFrame({"col": [1.0, 2.0]})

    with pytest.raises(
        ValueError,
        match=r"\{'OBS', 'STD'\}|\{'STD', 'OBS'\}"
        r" should be present in DataFrame columns,\s+"
        r"missing: (\{'OBS', 'STD'\}|\{'STD', 'OBS'\})",
    ):
        snake_oil_default_storage._validate_measured_data(frame_without_obs_and_std)
