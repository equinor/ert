import logging
import shutil

import pandas as pd
import pytest
from pandas.core.base import PandasObject

from ert._c_wrappers.enkf import EnKFMain, ResConfig
from ert._c_wrappers.enkf.export import SummaryCollector
from ert.libres_facade import LibresFacade


@pytest.fixture
def snake_oil_data(source_root, tmp_path, monkeypatch):
    shutil.copytree(
        source_root / "test-data/local/snake_oil", tmp_path, dirs_exist_ok=True
    )
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def facade(snake_oil_data):
    config_file = "snake_oil.ert"

    rc = ResConfig(user_config_file=config_file)
    rc.convertToCReference(None)
    ert = EnKFMain(rc)
    facade = LibresFacade(ert)
    return facade


def test_keyword_type_checks(facade):
    assert facade.is_gen_data_key("SNAKE_OIL_GPR_DIFF@199")
    assert facade.is_summary_key("BPR:1,3,8")
    assert facade.is_gen_kw_key("SNAKE_OIL_PARAM:BPR_138_PERSISTENCE")


def test_keyword_type_checks_missing_key(facade):
    assert not facade.is_gen_data_key("nokey")
    assert not facade.is_summary_key("nokey")
    assert not facade.is_gen_kw_key("nokey")


def test_data_fetching(facade):
    data = [
        facade.gather_gen_data_data("default_0", "SNAKE_OIL_GPR_DIFF@199"),
        facade.gather_summary_data("default_0", "BPR:1,3,8"),
        facade.gather_gen_kw_data("default_0", "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE"),
    ]

    for dataframe in data:
        assert isinstance(dataframe, PandasObject)
        assert not dataframe.empty


def test_data_fetching_missing_case(facade):
    data = [
        facade.gather_gen_data_data("nocase", "SNAKE_OIL_GPR_DIFF@199"),
        facade.gather_summary_data("nocase", "BPR:1,3,8"),
        facade.gather_gen_kw_data("nocase", "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE"),
    ]

    for dataframe in data:
        assert isinstance(dataframe, PandasObject)
        assert dataframe.empty


def test_data_fetching_missing_key(facade):
    data = [
        facade.gather_gen_data_data("default_0", "nokey"),
        facade.gather_summary_data("default_0", "nokey"),
        facade.gather_gen_kw_data("default_0", "nokey"),
    ]

    for dataframe in data:
        assert isinstance(dataframe, PandasObject)
        assert dataframe.empty


def test_cases_list(facade):
    cases = facade.cases()
    assert ["default_0", "default_1"] == cases


def test_case_has_data(facade):
    assert facade.case_has_data("default_0")
    assert not facade.case_has_data("default")


def test_all_data_type_keys(facade):
    keys = facade.all_data_type_keys()

    expected = [
        "BPR:1,3,8",
        "BPR:445",
        "BPR:5,5,5",
        "BPR:721",
        "FGIP",
        "FGIPH",
        "FGOR",
        "FGORH",
        "FGPR",
        "FGPRH",
        "FGPT",
        "FGPTH",
        "FOIP",
        "FOIPH",
        "FOPR",
        "FOPRH",
        "FOPT",
        "FOPTH",
        "FWCT",
        "FWCTH",
        "FWIP",
        "FWIPH",
        "FWPR",
        "FWPRH",
        "FWPT",
        "FWPTH",
        "WGOR:OP1",
        "WGOR:OP2",
        "WGORH:OP1",
        "WGORH:OP2",
        "WGPR:OP1",
        "WGPR:OP2",
        "WGPRH:OP1",
        "WGPRH:OP2",
        "WOPR:OP1",
        "WOPR:OP2",
        "WOPRH:OP1",
        "WOPRH:OP2",
        "WWCT:OP1",
        "WWCT:OP2",
        "WWCTH:OP1",
        "WWCTH:OP2",
        "WWPR:OP1",
        "WWPR:OP2",
        "WWPRH:OP1",
        "WWPRH:OP2",
        "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE",
        "SNAKE_OIL_PARAM:BPR_555_PERSISTENCE",
        "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE",
        "SNAKE_OIL_PARAM:OP1_OCTAVES",
        "SNAKE_OIL_PARAM:OP1_OFFSET",
        "SNAKE_OIL_PARAM:OP1_PERSISTENCE",
        "SNAKE_OIL_PARAM:OP2_DIVERGENCE_SCALE",
        "SNAKE_OIL_PARAM:OP2_OCTAVES",
        "SNAKE_OIL_PARAM:OP2_OFFSET",
        "SNAKE_OIL_PARAM:OP2_PERSISTENCE",
        "SNAKE_OIL_GPR_DIFF@199",
        "SNAKE_OIL_OPR_DIFF@199",
        "SNAKE_OIL_WPR_DIFF@199",
    ]

    assert expected == keys


def test_observation_keys(facade):
    expected_obs = {
        "FOPR": ["FOPR"],
        "WOPR:OP1": [
            "WOPR_OP1_108",
            "WOPR_OP1_190",
            "WOPR_OP1_144",
            "WOPR_OP1_9",
            "WOPR_OP1_72",
            "WOPR_OP1_36",
        ],
        "SNAKE_OIL_WPR_DIFF@199": ["WPR_DIFF_1"],
    }

    for key in facade.all_data_type_keys():
        obs_keys = facade.observation_keys(key)
        assert expected_obs.get(key, []) == obs_keys


def test_observation_keys_missing_key(facade):
    obs_keys = facade.observation_keys("nokey")
    assert [] == obs_keys


def test_case_refcase_data(facade):
    data = facade.refcase_data("FOPR")
    assert isinstance(data, PandasObject)


def test_case_refcase_data_missing_key(facade):
    data = facade.refcase_data("nokey")
    assert isinstance(data, PandasObject)


def test_case_history_data(facade):
    data = facade.history_data("FOPRH")
    assert isinstance(data, PandasObject)

    data = facade.history_data("FOPRH", case="default_1")
    assert isinstance(data, PandasObject)

    data = facade.history_data("WOPRH:OP1")
    assert isinstance(data, PandasObject)


def test_case_history_data_missing_key(facade):
    data = facade.history_data("nokey")
    assert isinstance(data, PandasObject)


def _do_verify_indices_and_values(data):
    # Verify indices
    assert data.columns.name == "Realization"
    assert all(data.columns == range(25))
    assert data.index.name == "Date"
    assert all(data.index == pd.date_range("2010-01-10", periods=200, freq="10D"))

    # Verify selected datapoints
    assert data.iloc[0][0] == pytest.approx(0.118963, abs=1e-6)  # top-left
    assert data.iloc[199][0] == pytest.approx(0.133601, abs=1e-6)  # bottom-left
    assert data.iloc[4][9] == pytest.approx(
        0.178028, abs=1e-6
    )  # somewhere in the middle
    # bottom-right 5 entries in col
    assert data.iloc[-5:][24].values == pytest.approx(
        [0.143714, 0.142230, 0.140191, 0.140143, 0.139711], abs=1e-6
    )


def test_summary_data_verify_indices_and_values(caplog, facade):
    with caplog.at_level(logging.WARNING):
        _do_verify_indices_and_values(facade.gather_summary_data("default_0", "FOPR"))
        assert "contains duplicate timestamps" not in caplog.text


_orig_loadAllSummaryData = SummaryCollector.loadAllSummaryData


def _add_duplicate_row(*args, **kwargs):
    df = _orig_loadAllSummaryData(*args, **kwargs)
    # Append copy of last date to each realization
    idx = pd.MultiIndex.from_tuples(
        [(i, df.loc[i].iloc[-1].name) for i in df.index.levels[0]]
    )
    df_new = pd.DataFrame(0, index=idx, columns=df.columns)
    return pd.concat([df, df_new]).sort_index()


def test_summary_data_verify_remove_duplicates(caplog, facade, monkeypatch):
    monkeypatch.setattr(SummaryCollector, "loadAllSummaryData", _add_duplicate_row)
    with caplog.at_level(logging.WARNING):
        _do_verify_indices_and_values(facade.gather_summary_data("default_0", "FOPR"))
        assert "contains duplicate timestamps" in caplog.text
