import os
import pathlib
import shutil
import time
from distutils.dir_util import copy_tree

import numpy as np
import pytest
from ecl.summary import EclSum
from ert.data import loader, MeasuredData
from ert_shared.libres_facade import LibresFacade
from res.enkf import EnKFMain, ResConfig
from res.enkf.export import SummaryObservationCollector

from ...utils import SOURCE_DIR

test_data_root = SOURCE_DIR / "test-data" / "local"


@pytest.fixture()
def copy_snake_oil(tmpdir):
    test_data_dir = os.path.join(test_data_root, "snake_oil")
    copy_tree(test_data_dir, str(tmpdir))
    with tmpdir.as_cwd():
        yield


@pytest.fixture()
def facade_snake_oil(copy_snake_oil):
    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)
    yield LibresFacade(ert)


def test_history_obs(monkeypatch, facade_snake_oil):

    fopr = MeasuredData(facade_snake_oil, ["FOPR"])
    fopr.remove_inactive_observations()

    assert all(
        fopr.data.columns.get_level_values("data_index").values == list(range(200))
    )


def test_summary_obs(monkeypatch, facade_snake_oil):
    summary_obs = MeasuredData(facade_snake_oil, ["WOPR_OP1_72"])
    summary_obs.remove_inactive_observations()
    assert all(summary_obs.data.columns.get_level_values("data_index").values == [71])
    # Only one observation, we check the key_index is what we expect:
    assert summary_obs.data.columns.get_level_values("key_index").values[
        0
    ] == np.datetime64("2011-12-21")


@pytest.mark.integration_test
def test_summary_obs_runtime(monkeypatch, copy_snake_oil):
    """
    This is mostly a regression test, as reading SUMMARY_OBS was very slow when using
    SUMMARY_OBSERVATION and not HISTORY_OBSERVATION where multiple observations
    were pointing to the same response. To simulate that we load the same observations
    though individual points, and also in one go. To avoid this test being flaky the
    we assert on the difference in runtime. The difference in runtime we assert on is
    set to 10x though it should be around 2x
    """

    obs_file = pathlib.Path.cwd() / "observations" / "observations.txt"
    with obs_file.open(mode="a") as fin:
        fin.write(create_summary_observation())

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)

    facade = LibresFacade(ert)

    start_time = time.time()
    foprh = MeasuredData(facade, [f"FOPR_{restart}" for restart in range(1, 201)])
    summary_obs_time = time.time() - start_time

    start_time = time.time()
    fopr = MeasuredData(facade, ["FOPR"])
    history_obs_time = time.time() - start_time

    assert (
        fopr.data.columns.get_level_values("data_index").values.tolist()
        == foprh.data.columns.get_level_values("data_index").values.tolist()
    )

    result = foprh.get_simulated_data().values == fopr.get_simulated_data().values
    assert np.logical_and.reduce(result).all()
    assert summary_obs_time < 10 * history_obs_time


@pytest.mark.parametrize("formatted_date", ["2015-06-23", "23/06/2015", "23.06.2015"])
def test_summary_obs_last_entry(monkeypatch, copy_snake_oil, formatted_date):

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

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)

    facade = LibresFacade(ert)

    foprh = MeasuredData(facade, ["LAST_DATE"])
    assert foprh.data.loc[["OBS", "STD"]].values.flatten().tolist() == [10.0, 0.1]
    assert list(foprh.data.columns.get_level_values("key_index").values) == [
        np.datetime64("2015-06-23")
    ]


@pytest.mark.integration_test
def test_gen_obs_runtime(monkeypatch, copy_snake_oil, snapshot):
    obs_file = pathlib.Path.cwd() / "observations" / "observations.txt"
    with obs_file.open(mode="a") as fin:
        fin.write(create_general_observation())

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)

    facade = LibresFacade(ert)

    df = MeasuredData(facade, [f"CUSTOM_DIFF_{restart}" for restart in range(500)])

    snapshot.assert_match(df.data.to_csv(), "snake_oil_gendata_output.csv")


@pytest.mark.parametrize("formatted_date", ["2015-06-23", "23/06/2015", "23.06.2015"])
def test_gen_obs(monkeypatch, facade_snake_oil, formatted_date):
    df = MeasuredData(facade_snake_oil, ["WPR_DIFF_1"])
    df.remove_inactive_observations()

    assert all(
        df.data.columns.get_level_values("data_index").values == [400, 800, 1200, 1800]
    )
    assert all(
        df.data.columns.get_level_values("key_index").values == [400, 800, 1200, 1800]
    )


@pytest.mark.parametrize("formatted_date", ["2010-01-10", "10/1/2010", "10.1.2010"])
def test_block_obs(monkeypatch, tmpdir, formatted_date):
    """
    This test causes util_abort on some runs, so it will not be run by default
    as it is too flaky. I have chosen to leave it here as it could be useful when
    debugging. To run the test, run an ensemble_experiment on the snake_oil_field
    case to create a storage with BLOCK_OBS.
    """
    with tmpdir.as_cwd():
        test_data_dir = pathlib.Path(test_data_root) / "snake_oil_field"
        if not (test_data_dir / "storage").exists():
            pytest.skip()
        else:
            shutil.copytree(test_data_dir, "test_data")
            os.chdir("test_data")

            block_obs = (
                "\n"
                "BLOCK_OBSERVATION RFT_2006\n"
                "{\n"
                "   FIELD = PRESSURE;\n"
                f"   DATE  = {formatted_date};\n"
                "   SOURCE = SUMMARY;\n"
                "   OBS P1 { I = 5;  J = 5;  K = 5;   VALUE = 100;  ERROR = 5; };\n"
                "   OBS P2 { I = 1;  J = 3;  K = 8;   VALUE = 50;  ERROR = 2; };\n"
                "};\n"
            )
            obs_file = pathlib.Path.cwd() / "observations" / "observations.txt"
            with obs_file.open(mode="a") as fin:
                fin.write(block_obs)

            res_config = ResConfig("snake_oil.ert")
            ert = EnKFMain(res_config)
            facade = LibresFacade(ert)

            df = MeasuredData(facade, ["RFT_2006"])
            df.remove_inactive_observations()
            assert all(df.data.columns.get_level_values("data_index").values == [0, 1])
            assert all(df.data.columns.get_level_values("key_index").values == [0, 1])


def test_gen_obs_and_summary(monkeypatch, facade_snake_oil):
    df = MeasuredData(facade_snake_oil, ["WPR_DIFF_1", "WOPR_OP1_9"])
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


def test_gen_obs_and_summary_index_range(monkeypatch, facade_snake_oil):
    df = MeasuredData(facade_snake_oil, ["WPR_DIFF_1", "FOPR"], [[800], [10]])
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
        ("FOPR", r"No response loaded for observation keys: \['FOPR'\]"),
        ("WPR_DIFF_1", "No response loaded for observation key: WPR_DIFF_1"),
    ],
)
@pytest.mark.usefixtures("copy_snake_oil")
def test_no_storage(monkeypatch, obs_key, expected_msg):
    shutil.rmtree("storage")
    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)

    facade = LibresFacade(ert)
    with pytest.raises(
        loader.ResponseError,
        match=expected_msg,
    ):
        MeasuredData(facade, [obs_key])


@pytest.mark.parametrize("obs_key", ["FOPR", "WPR_DIFF_1"])
@pytest.mark.usefixtures("copy_snake_oil")
def test_no_storage_obs_only(monkeypatch, obs_key):
    shutil.rmtree("storage")
    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)

    facade = LibresFacade(ert)
    md = MeasuredData(facade, [obs_key], load_data=False)
    assert set(md.data.columns.get_level_values(0)) == {obs_key}


@pytest.mark.usefixtures("copy_snake_oil")
def test_summary_collector():
    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)
    summary = EclSum("refcase/SNAKE_OIL_FIELD.UNSMRY")
    data = SummaryObservationCollector.loadObservationData(ert, "default_0")

    assert (
        data["FOPR"].values.tolist()
        == summary.numpy_vector("FOPRH", report_only=True).tolist()
    )


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


def test_all_measured_snapshot(snapshot, facade_snake_oil):
    """
    While there is no guarantee that this snapshot is 100% correct, it does represent
    the current state of loading from storage for the snake_oil case.
    """
    obs_keys = facade_snake_oil.get_matching_wildcards()("*").strings
    measured_data = MeasuredData(facade_snake_oil, obs_keys)
    snapshot.assert_match(measured_data.data.to_csv(), "snake_oil_measured_output.csv")


def test_active_realizations(facade_snake_oil):
    current_case_name = facade_snake_oil.get_current_case_name()
    active_realizations = facade_snake_oil.get_active_realizations(current_case_name)
    assert len(active_realizations) == 25
    assert active_realizations == list(range(25))


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
