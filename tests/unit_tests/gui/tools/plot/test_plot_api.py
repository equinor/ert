import httpx
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


def test_key_def_structure(api):
    key_defs = api.all_data_type_keys()
    fopr = next(x for x in key_defs if x["key"] == "FOPR")
    fopr_expected = {
        "dimensionality": 2,
        "index_type": "VALUE",
        "key": "FOPR",
        "metadata": {"data_origin": "Summary"},
        "observations": True,
        "log_scale": False,
    }
    assert fopr == fopr_expected

    bpr = next(x for x in key_defs if x["key"] == "BPR:1,3,8")
    bpr_expected = {
        "dimensionality": 2,
        "index_type": "VALUE",
        "key": "BPR:1,3,8",
        "metadata": {"data_origin": "Summary"},
        "observations": False,
        "log_scale": False,
    }
    assert bpr == bpr_expected

    bpr_parameter = next(
        x for x in key_defs if x["key"] == "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE"
    )
    bpr_parameter_expected = {
        "dimensionality": 1,
        "index_type": None,
        "key": "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE",
        "metadata": {"data_origin": "GEN_KW"},
        "observations": False,
        "log_scale": False,
    }
    assert bpr_parameter == bpr_parameter_expected


def test_case_structure(api):
    cases = [case["name"] for case in api.get_all_cases_not_running()]
    hidden_case = [
        case["name"] for case in api.get_all_cases_not_running() if case["hidden"]
    ]
    expected = ["ensemble_1", ".ensemble_2", "default_0", "default_1"]

    assert cases == expected
    assert hidden_case == [".ensemble_2"]


def test_can_load_data_and_observations(api):
    keys = {
        "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE": {
            "key": "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE",
            "observations": False,
            "userdata": {"data_origin": "GEN_KW"},
        },
        "BPR:1,3,8": {
            "key": "BPR:1,3,8",
            "userdata": {"data_origin": "Summary"},
            "observations": False,
        },
        "FOPR": {
            "key": "FOPR",
            "userdata": {"data_origin": "Summary"},
            "observations": True,
        },
    }

    case_name = "default_0"
    for key, value in keys.items():
        observations = value["observations"]
        if observations:
            obs_data = api.observations_for_key(case_name, key)
            assert not obs_data.empty
        data = api.data_for_key(case_name, key)
        assert not data.empty


def test_all_data_type_keys(api):
    keys = [e["key"] for e in api.all_data_type_keys()]
    assert keys == [
        "BPR:1,3,8",
        "FOPR",
        "SNAKE_OIL_WPR_DIFF@199",
        "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE",
        "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE",
        "WOPPER",
        "I_AM_A_PARAM",
    ]


def test_load_history_data(api):
    df = api.history_data(case="default_0", key="FOPR")
    assert_frame_equal(
        df, pd.DataFrame({1: [0.2, 0.2, 1.2], 3: [1.0, 1.1, 1.2], 4: [1.0, 1.1, 1.3]})
    )


def test_plot_api_request_errors(api, mocker):
    # Mock the experiment name to be something unexpected
    mocker.patch(
        "ert.gui.tools.plot.plot_api.PlotApi._get_experiments",
        return_value=[{"id": "mocked"}],
    )
    with pytest.raises(httpx.RequestError):
        api.all_data_type_keys()

    case_name = "default_0"
    with pytest.raises(httpx.RequestError):
        api.observations_for_key(case_name, "should_not_be_there")

    with pytest.raises(httpx.RequestError):
        api.data_for_key(case_name, "should_not_be_there")
