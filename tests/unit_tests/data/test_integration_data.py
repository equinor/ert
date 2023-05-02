import pathlib

import numpy as np
import pytest

from ert.data import MeasuredData
from ert.data._measured_data import ResponseError
from ert.libres_facade import LibresFacade
from ert.storage import open_storage


@pytest.fixture()
def facade_snake_oil(snake_oil_case_storage):
    yield LibresFacade(snake_oil_case_storage)


@pytest.fixture
def default_ensemble(snake_oil_case_storage):
    with open_storage(snake_oil_case_storage.ert_config.ens_path) as storage:
        yield storage.get_ensemble_by_name("default_0")


@pytest.fixture
def create_measured_data(snake_oil_case_storage, snake_oil_default_storage):
    def func(*args, **kwargs):
        return MeasuredData(
            LibresFacade(snake_oil_case_storage),
            snake_oil_default_storage,
            *args,
            **kwargs,
        )

    return func


def test_history_obs(create_measured_data):
    fopr = create_measured_data(["FOPR"])
    fopr.remove_inactive_observations()

    assert all(
        fopr.data.columns.get_level_values("data_index").values == list(range(200))
    )


def test_summary_obs(create_measured_data):
    summary_obs = create_measured_data(["WOPR_OP1_72"])
    summary_obs.remove_inactive_observations()
    assert all(summary_obs.data.columns.get_level_values("data_index").values == [71])
    # Only one observation, we check the key_index is what we expect:
    assert summary_obs.data.columns.get_level_values("key_index").values[
        0
    ] == np.datetime64("2011-12-21")


@pytest.mark.filterwarnings("ignore::ert.parsing.ConfigWarning")
@pytest.mark.usefixtures("copy_snake_oil_case_storage")
@pytest.mark.parametrize("formatted_date", ["2015-06-23", "23/06/2015"])
def test_summary_obs_last_entry(formatted_date):
    obs_file = pathlib.Path.cwd() / "observations" / "observations.txt"
    with obs_file.open(mode="w") as fin:
        fin.write(
            "\n"
            "SUMMARY_OBSERVATION LAST_DATE\n"
            "{\n"
            "   VALUE   = 10;\n"
            "   ERROR   = 0.1;\n"
            f"   DATE    = {formatted_date};\n"
            "   KEY     = FOPR;\n"
            "};\n"
        )

    facade = LibresFacade.from_config_file("snake_oil.ert")
    storage = open_storage(facade.enspath)
    ensemble = storage.get_ensemble_by_name("default_0")

    foprh = MeasuredData(facade, ensemble, ["LAST_DATE"])

    assert foprh.data.loc[["OBS", "STD"]].values.flatten().tolist() == [10.0, 0.1]
    assert list(foprh.data.columns.get_level_values("key_index").values) == [
        np.datetime64("2015-06-23")
    ]


@pytest.mark.integration_test
def test_gen_obs_runtime(snapshot, create_measured_data):
    obs_file = pathlib.Path.cwd() / "observations" / "observations.txt"
    with obs_file.open(mode="a") as fin:
        fin.write(create_general_observation())

    facade = LibresFacade.from_config_file("snake_oil.ert")
    storage = open_storage(facade.enspath)
    ensemble = storage.get_ensemble_by_name("default_0")

    df = MeasuredData(
        facade, ensemble, [f"CUSTOM_DIFF_{restart}" for restart in range(500)]
    )

    snapshot.assert_match(df.data.to_csv(), "snake_oil_gendata_output.csv")


def test_gen_obs(create_measured_data):
    df = create_measured_data(["WPR_DIFF_1"])
    df.remove_inactive_observations()

    assert all(
        df.data.columns.get_level_values("data_index").values == [400, 800, 1200, 1800]
    )
    assert all(
        df.data.columns.get_level_values("key_index").values == [400, 800, 1200, 1800]
    )


def test_gen_obs_and_summary(create_measured_data):
    df = create_measured_data(["WPR_DIFF_1", "WOPR_OP1_9"])
    df.remove_inactive_observations()

    assert df.data.columns.get_level_values(0).to_list() == [
        "WPR_DIFF_1",
        "WPR_DIFF_1",
        "WPR_DIFF_1",
        "WPR_DIFF_1",
        "WOPR_OP1_9",
    ]
    assert df.data.columns.get_level_values("data_index").to_list() == [
        400,
        800,
        1200,
        1800,
        8,
    ]


def test_gen_obs_and_summary_index_range(create_measured_data):
    df = create_measured_data(["WPR_DIFF_1", "FOPR"], [[800], [10]])
    df.remove_inactive_observations()

    assert df.data.columns.get_level_values(0).to_list() == [
        "WPR_DIFF_1",
        "FOPR",
    ]
    assert df.data.columns.get_level_values("data_index").to_list() == [
        800,
        10,
    ]
    assert df.data.loc["OBS"].values == pytest.approx([0.1, 0.23281], abs=0.00001)
    assert df.data.loc["STD"].values == pytest.approx([0.2, 0.1])


@pytest.mark.parametrize(
    "obs_key, expected_msg",
    [
        ("FOPR", r"No response loaded for observation key: FOPR"),
        ("WPR_DIFF_1", "No response loaded for observation key: WPR_DIFF_1"),
    ],
)
def test_no_storage(obs_key, expected_msg, facade_snake_oil, storage):
    ensemble = storage.create_experiment().create_ensemble(
        name="empty", ensemble_size=facade_snake_oil.get_ensemble_size()
    )

    with pytest.raises(
        ResponseError,
        match=expected_msg,
    ):
        MeasuredData(facade_snake_oil, ensemble, [obs_key])


def create_summary_observation():
    observations = ""
    values = np.random.uniform(0, 1.5, 200)
    errors = values * 0.1
    for restart, (value, error) in enumerate(zip(values, errors)):
        restart += 1
        observations += f"""
    \nSUMMARY_OBSERVATION FOPR_{restart}
{{
    VALUE   = {value};
    ERROR   = {error};
    RESTART = {restart};
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


def test_all_measured_snapshot(snapshot, facade_snake_oil, create_measured_data):
    """
    While there is no guarantee that this snapshot is 100% correct, it does represent
    the current state of loading from storage for the snake_oil case.
    """
    obs_keys = facade_snake_oil.get_matching_wildcards()("*")
    measured_data = create_measured_data(obs_keys)
    snapshot.assert_match(measured_data.data.to_csv(), "snake_oil_measured_output.csv")


def test_active_realizations(facade_snake_oil, default_ensemble):
    active_realizations = facade_snake_oil.get_active_realizations(default_ensemble)
    assert len(active_realizations) == 5
    assert active_realizations == list(range(5))


def test_get_data_keys(facade_snake_oil):
    summary_keys = set(facade_snake_oil.get_summary_keys())
    for key in summary_keys:
        assert facade_snake_oil.is_summary_key(key)

    gen_data_keys = set(facade_snake_oil.get_gen_data_keys())
    for key in gen_data_keys:
        assert facade_snake_oil.is_gen_data_key(key)

    gen_kw_keys = set(facade_snake_oil.gen_kw_keys())
    for key in gen_kw_keys:
        assert facade_snake_oil.is_gen_kw_key(key)

    all_keys = set(facade_snake_oil.all_data_type_keys())
    diff = all_keys.difference(summary_keys, gen_data_keys, gen_kw_keys)

    assert len(diff) == 0
