import logging
from textwrap import dedent

import pandas as pd
import pytest
from pandas.core.base import PandasObject

from ert.libres_facade import LibresFacade


@pytest.fixture
def facade(snake_oil_case):
    return LibresFacade(snake_oil_case)


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


def test_summary_keys(facade):
    assert len(facade.get_summary_keys()) == 46
    assert "FOPT" in facade.get_summary_keys()


def test_gen_data_keys(facade):
    assert len(facade.get_gen_data_keys()) == 3
    assert "SNAKE_OIL_WPR_DIFF@199" in facade.get_gen_data_keys()


def test_gen_kw_keys(facade):
    assert len(facade.gen_kw_keys()) == 10
    assert "SNAKE_OIL_PARAM:BPR_555_PERSISTENCE" in facade.gen_kw_keys()


def test_gen_kw_priors(facade):
    priors = facade.gen_kw_priors()
    assert len(priors["SNAKE_OIL_PARAM"]) == 10
    assert {
        "key": "OP1_PERSISTENCE",
        "function": "UNIFORM",
        "parameters": {"MIN": 0.01, "MAX": 0.4},
    } in priors["SNAKE_OIL_PARAM"]


def test_summary_collector(monkeypatch, facade):
    monkeypatch.setenv("TZ", "CET")  # The ert_statoil case was generated in CET

    data = facade.load_all_summary_data("default_0")

    assert pytest.approx(data["WWCT:OP2"][0]["2010-01-10"], rel=1e-5) == 0.385549
    assert pytest.approx(data["WWCT:OP2"][24]["2010-01-10"]) == 0.498331

    assert pytest.approx(data["FOPR"][0]["2010-01-10"], rel=1e-5) == 0.118963
    assert pytest.approx(data["FOPR"][0]["2015-06-23"], rel=1e-5) == 0.133601

    # pylint: disable=pointless-statement
    # realization 20:
    data.loc[20]

    with pytest.raises(KeyError):
        # realization 60:
        data.loc[60]

    data = facade.load_all_summary_data("default_0", ["WWCT:OP1", "WWCT:OP2"])

    assert pytest.approx(data["WWCT:OP1"][0]["2010-01-10"]) == 0.352953
    assert pytest.approx(data["WWCT:OP2"][0]["2010-01-10"], rel=1e-5) == 0.385549

    with pytest.raises(KeyError):
        data["FOPR"]

    realization_index = 10
    data = facade.load_all_summary_data(
        "default_0",
        ["WWCT:OP1", "WWCT:OP2"],
        realization_index=realization_index,
    )

    assert data.index.levels[0] == [realization_index]
    assert len(data.index.levels[1]) == 200
    assert list(data.columns) == ["WWCT:OP1", "WWCT:OP2"]

    non_existing_realization_index = 150
    with pytest.raises(IndexError):
        data = facade.load_all_summary_data(
            "default_0",
            ["WWCT:OP1", "WWCT:OP2"],
            realization_index=non_existing_realization_index,
        )


def test_misfit_collector(facade):
    data = facade.load_all_misfit_data("default_0")

    assert pytest.approx(data["MISFIT:FOPR"][0]) == 738.735586
    assert pytest.approx(data["MISFIT:FOPR"][24]) == 1260.086789

    assert pytest.approx(data["MISFIT:TOTAL"][0]) == 767.008457
    assert pytest.approx(data["MISFIT:TOTAL"][24]) == 1359.172803

    # pylint: disable=pointless-statement
    # realization 20:
    data.loc[20]

    with pytest.raises(KeyError):
        # realization 60:
        data.loc[60]


def test_gen_kw_collector(facade):
    data = facade.load_all_gen_kw_data("default_0")

    assert (
        pytest.approx(data["SNAKE_OIL_PARAM:OP1_PERSISTENCE"][0], rel=1e-5) == 0.047517
    )
    assert (
        pytest.approx(data["SNAKE_OIL_PARAM:OP1_PERSISTENCE"][24], rel=1e-5) == 0.160907
    )

    assert pytest.approx(data["SNAKE_OIL_PARAM:OP1_OFFSET"][0], rel=1e-5) == 0.054539
    assert pytest.approx(data["SNAKE_OIL_PARAM:OP1_OFFSET"][12], rel=1e-5) == 0.057807

    # pylint: disable=pointless-statement
    # realization 20:
    data.loc[20]

    with pytest.raises(KeyError):
        # realization 60:
        data.loc[60]

    data = facade.load_all_gen_kw_data(
        "default_0",
        ["SNAKE_OIL_PARAM:OP1_PERSISTENCE", "SNAKE_OIL_PARAM:OP1_OFFSET"],
    )

    assert (
        pytest.approx(data["SNAKE_OIL_PARAM:OP1_PERSISTENCE"][0], rel=1e-5) == 0.047517
    )
    assert pytest.approx(data["SNAKE_OIL_PARAM:OP1_OFFSET"][0], rel=1e-5) == 0.054539

    with pytest.raises(KeyError):
        data["SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE"]

    realization_index = 10
    data = facade.load_all_gen_kw_data(
        "default_0",
        ["SNAKE_OIL_PARAM:OP1_PERSISTENCE"],
        realization_index=realization_index,
    )

    assert data.index == [realization_index]
    assert len(data.index) == 1
    assert list(data.columns) == ["SNAKE_OIL_PARAM:OP1_PERSISTENCE"]
    assert (
        pytest.approx(data["SNAKE_OIL_PARAM:OP1_PERSISTENCE"][10], rel=1e-5) == 0.282923
    )

    non_existing_realization_index = 150
    with pytest.raises(IndexError):
        data = facade.load_all_gen_kw_data(
            "default_0",
            ["SNAKE_OIL_PARAM:OP1_PERSISTENCE"],
            realization_index=non_existing_realization_index,
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_data_report_steps():
    with open("config_file.ert", "w") as fout:
        # Write a minimal config file
        fout.write(
            dedent(
                """
        NUM_REALIZATIONS 1
        OBS_CONFIG observations
        TIME_MAP time_map
        GEN_DATA RESPONSE RESULT_FILE:result_%d.out REPORT_STEPS:0,1 INPUT_FORMAT:ASCII
        """
            )
        )
    with open("obs_data_0.txt", "w") as fout:
        fout.write("1.0 0.1")
    with open("obs_data_1.txt", "w") as fout:
        fout.write("2.0 0.1")

    with open("time_map", "w") as fout:
        fout.write("2014-09-10\n2017-02-05")

    with open("observations", "w") as fout:
        fout.write(
            dedent(
                """
        GENERAL_OBSERVATION OBS_0 {
            DATA       = RESPONSE;
            INDEX_LIST = 0;
            RESTART    = 0;
            OBS_FILE   = obs_data_0.txt;
        };

        GENERAL_OBSERVATION OBS_1 {
            DATA       = RESPONSE;
            INDEX_LIST = 0;
            RESTART    = 1;
            OBS_FILE   = obs_data_1.txt;
        };
        """
            )
        )
    facade = LibresFacade.from_config_file("config_file.ert")
    obs_key = facade.observation_keys("RESPONSE@0")
    assert obs_key == ["OBS_0"]

    obs_key = facade.observation_keys("RESPONSE@1")
    assert obs_key == ["OBS_1"]

    obs_key = facade.observation_keys("RESPONSE@2")
    assert obs_key == []

    obs_key = facade.observation_keys("NOT_A_KEY")
    assert obs_key == []


def test_gen_data_collector(facade):
    with pytest.raises(KeyError):
        facade.load_gen_data("default_0", "RFT_XX", 199)

    with pytest.raises(ValueError):
        facade.load_gen_data("default_0", "SNAKE_OIL_OPR_DIFF", 198)

    data1 = facade.load_gen_data("default_0", "SNAKE_OIL_OPR_DIFF", 199)

    assert pytest.approx(data1[0][0]) == -0.008206
    assert pytest.approx(data1[24][1]) == -0.119255
    assert pytest.approx(data1[24][1000]) == -0.258516

    realization_index = 10
    data1 = facade.load_gen_data(
        "default_0",
        "SNAKE_OIL_OPR_DIFF",
        199,
        realization_index=realization_index,
    )

    assert len(data1.index) == 2000
    assert list(data1.columns) == [realization_index]

    realization_index = 150
    with pytest.raises(IndexError):
        data1 = facade.load_gen_data(
            "default_0",
            "SNAKE_OIL_OPR_DIFF",
            199,
            realization_index=realization_index,
        )
