import gc
import unittest
from datetime import date, datetime
from textwrap import dedent
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
from pandas.testing import assert_frame_equal
from starlette.testclient import TestClient

from ert.config import GenKwConfig, SummaryConfig
from ert.config.gen_kw_config import TransformFunctionDefinition
from ert.config.parameter_config import ParameterMetadata
from ert.config.response_config import ResponseMetadata
from ert.dark_storage import common
from ert.dark_storage.app import app
from ert.gui.tools.plot.plot_api import PlotApi, PlotApiKeyDefinition
from ert.services import StorageService
from ert.storage import open_storage
from tests.ert.unit_tests.gui.tools.plot.conftest import MockResponse


@pytest.fixture(autouse=True)
def use_testclient(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(StorageService, "session", lambda project: client)

    def test_escape(s: str) -> str:
        """
        Workaround for issue with TestClient:
        https://github.com/encode/starlette/issues/1060
        """
        return quote(quote(quote(s, safe="")))

    PlotApi.escape = test_escape


def test_key_def_structure(api):
    key_defs = api.parameters_api_key_defs + api.responses_api_key_defs
    fopr = next(x for x in key_defs if x.key == "FOPR")
    fopr_expected = {
        "dimensionality": 2,
        "index_type": "VALUE",
        "key": "FOPR",
        "metadata": {"data_origin": "summary"},
        "observations": True,
        "log_scale": False,
        "parameter_metadata": None,
        "response_metadata": ResponseMetadata(
            response_type="summary",
            response_key="FOPR",
            filter_on={},
            finalized=True,
        ),
    }
    assert fopr == PlotApiKeyDefinition(**fopr_expected)

    bpr = next(x for x in key_defs if x.key == "BPR:1,3,8")
    bpr_expected = {
        "dimensionality": 2,
        "index_type": "VALUE",
        "key": "BPR:1,3,8",
        "metadata": {"data_origin": "summary"},
        "observations": False,
        "log_scale": False,
        "parameter_metadata": None,
        "response_metadata": ResponseMetadata(
            response_type="summary",
            response_key="BPR:1,3,8",
            filter_on={},
            finalized=True,
        ),
    }
    assert bpr == PlotApiKeyDefinition(**bpr_expected)

    bpr_parameter = next(
        x for x in key_defs if x.key == "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE"
    )
    bpr_parameter_expected = {
        "dimensionality": 1,
        "index_type": None,
        "key": "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE",
        "metadata": {"data_origin": "GEN_KW"},
        "observations": False,
        "log_scale": False,
        "parameter_metadata": ParameterMetadata(
            key="SNAKE_OIL_PARAM:BPR_138_PERSISTENCE",
            transformation="NORMAL",
            dimensionality=1,
            userdata={"data_origin": "GEN_KW"},
        ),
        "response_metadata": None,
    }
    assert bpr_parameter == PlotApiKeyDefinition(**bpr_parameter_expected)


def test_case_structure(api):
    ensembles = [ensemble.name for ensemble in api.get_all_ensembles()]
    hidden_case = [
        ensemble.name for ensemble in api.get_all_ensembles() if ensemble.hidden
    ]
    expected = [
        "ensemble_1",
        ".ensemble_2",
        "default_0",
        "default_1",
        "default_manyobs",
    ]

    assert ensembles == expected
    assert hidden_case == [".ensemble_2"]


def test_can_load_data_and_observations(api):
    responses = [("BPR:1,3,8", False), ("FOPR", True)]
    ensemble = next(x for x in api.get_all_ensembles() if x.name == "default_0")
    data = api.data_for_parameter(ensemble.id, "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE")
    assert not data.empty

    for key, has_observations in responses:
        if has_observations:
            obs_data = api.observations_for_key([ensemble.id], key)
            assert not obs_data.empty
        data = api.data_for_response(ensemble.id, key)
        assert not data.empty


def test_all_data_type_keys(api):
    keys = [e.key for e in (api.parameters_api_key_defs + api.responses_api_key_defs)]
    assert sorted(keys) == sorted(
        [
            "BPR:1,3,8",
            "FOPR",
            "SNAKE_OIL_WPR_DIFF@199",
            "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE",
            "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE",
            "WOPPER",
            "I_AM_A_PARAM",
        ]
    )


def test_load_history_data(api):
    ens_id = next(ens.id for ens in api.get_all_ensembles() if ens.name == "default_0")
    df = api.history_data(ensemble_ids=[ens_id], key="FOPR")
    assert_frame_equal(
        df, pd.DataFrame({1: [0.2, 0.2, 1.2], 3: [1.0, 1.1, 1.2], 4: [1.0, 1.1, 1.3]})
    )


def test_load_history_data_searches_until_history_found(api):
    ensemble_ids = [
        ens.id
        for ens in api.get_all_ensembles()
        if ens.name in {"no-history", "default_0"}
    ]
    df = api.history_data(ensemble_ids=ensemble_ids, key="FOPR")
    assert_frame_equal(
        df, pd.DataFrame({1: [0.2, 0.2, 1.2], 3: [1.0, 1.1, 1.2], 4: [1.0, 1.1, 1.3]})
    )


def test_load_history_data_returns_empty_frame_if_no_history(api):
    ensemble_ids = [
        ens.id
        for ens in api.get_all_ensembles()
        if ens.name in {"no-history", "still-no-history"}
    ]
    df = api.history_data(ensemble_ids=ensemble_ids, key="FOPR")
    assert_frame_equal(df, pd.DataFrame())


def test_plot_api_request_errors_all_data_type_keys(api, mocker):
    # Mock the experiment name to be something unexpected
    mocker.patch(
        "tests.ert.unit_tests.gui.tools.plot.conftest.mocked_requests_get",
        return_value=MockResponse(None, 404, text="error"),
    )

    with pytest.raises(httpx.RequestError):
        api.parameters_api_key_defs + api.responses_api_key_defs


def test_plot_api_request_errors(api):
    ensemble = next(x for x in api.get_all_ensembles() if x.name == "default_0")

    with pytest.raises(httpx.RequestError):
        api.observations_for_key([ensemble.id], "should_not_be_there")

    with pytest.raises(httpx.RequestError):
        api.data_for_response(ensemble.id, "should_not_be_there")


@pytest.fixture
def api_and_storage(monkeypatch, tmp_path):
    ens_path = tmp_path / "storage"
    with open_storage(ens_path, mode="w") as storage:
        monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")
        monkeypatch.setenv("ERT_STORAGE_ENS_PATH", str(storage.path))
        api = PlotApi(ens_path)
        yield api, storage
    if common._storage is not None:
        common._storage.close()
    common._storage = None
    gc.collect()


def test_plot_api_handles_urlescape(api_and_storage):
    api, storage = api_and_storage
    key = "WBHP:46/3-7S"
    date = datetime(year=2024, month=10, day=4)
    experiment = storage.create_experiment(
        parameters=[],
        responses=[
            SummaryConfig(
                name="summary",
                input_files=["CASE.UNSMRY", "CASE.SMSPEC"],
                keys=[key],
            )
        ],
        observations={
            "summary": pl.DataFrame(
                {
                    "response_key": key,
                    "observation_key": "sumobs",
                    "time": pl.Series([date]).dt.cast_time_unit("ms"),
                    "observations": pl.Series([1.0], dtype=pl.Float32),
                    "std": pl.Series([1.0], dtype=pl.Float32),
                }
            )
        },
    )
    ensemble = experiment.create_ensemble(ensemble_size=1, name="ensemble")
    assert api.data_for_response(str(ensemble.id), key).empty
    df = pl.DataFrame(
        {
            "response_key": [key],
            "time": [pl.Series([date]).dt.cast_time_unit("ms")],
            "values": [pl.Series([1.0], dtype=pl.Float32)],
        }
    )
    df = df.explode("values", "time")
    ensemble.save_response(
        "summary",
        df,
        0,
    )
    assert api.data_for_response(str(ensemble.id), key).to_csv() == dedent(
        """\
        Realization,2024-10-04
        0,1.0
        """
    )
    assert api.observations_for_key([str(ensemble.id)], key).to_csv() == dedent(
        """\
        ,0
        STD,1.0
        OBS,1.0
        key_index,2024-10-04 00:00:00
        """
    )


def test_plot_api_handles_empty_gen_kw(api_and_storage):
    api, storage = api_and_storage
    key = "gen_kw"
    name = "<poro>"
    experiment = storage.create_experiment(
        parameters=[
            GenKwConfig(
                name=key,
                forward_init=False,
                update=False,
                transform_function_definitions=[
                    TransformFunctionDefinition(
                        name=name, param_name="NORMAL", values=[0, 0.1]
                    )
                ],
            ),
        ],
        responses=[],
        observations={},
    )
    ensemble = storage.create_ensemble(experiment.id, ensemble_size=10)
    assert api.data_for_parameter(str(ensemble.id), key).empty
    ensemble.save_parameters(
        key,
        realization=None,
        dataset=pl.DataFrame(
            {
                name: [1.0],
                "realization": 1,
            }
        ),
    )
    assert api.data_for_parameter(
        str(ensemble.id), key + ":" + name
    ).to_csv() == dedent(
        """\
        Realization,0
        1,0.1
        """
    )


def test_plot_api_handles_non_existant_gen_kw(api_and_storage):
    api, storage = api_and_storage
    experiment = storage.create_experiment(
        parameters=[
            GenKwConfig(
                name="gen_kw",
                forward_init=False,
                update=False,
                transform_function_definitions=[],
            ),
        ],
        responses=[],
        observations={},
    )
    ensemble = storage.create_ensemble(experiment.id, ensemble_size=10)
    ensemble.save_parameters(
        "gen_kw",
        1,
        xr.Dataset(
            {
                "values": ("names", [1.0]),
                "names": ["key"],
            }
        ),
    )
    assert api.data_for_parameter(str(ensemble.id), "gen_kw").empty
    assert api.data_for_parameter(str(ensemble.id), "gen_kw:does_not_exist").empty


def test_plot_api_handles_colons_in_parameter_keys(api_and_storage):
    api, storage = api_and_storage
    experiment = storage.create_experiment(
        parameters=[
            GenKwConfig(
                name="group",
                forward_init=False,
                update=False,
                transform_function_definitions=[
                    TransformFunctionDefinition(
                        name="subgroup:1:2:2", param_name="RAW", values=[]
                    ),
                ],
            ),
        ],
        responses=[],
        observations={},
    )
    ensemble = storage.create_ensemble(experiment.id, ensemble_size=10)
    ensemble.save_parameters(
        "group",
        0,
        pl.DataFrame(
            {
                "subgroup:1:2:2": pl.Series([10], dtype=pl.Float32),
                "realization": pl.Series([0], dtype=pl.Int32),
            }
        ),
    )
    test = api.data_for_parameter(str(ensemble.id), "group:subgroup:1:2:2")
    assert test.to_numpy() == np.array([[10]])


def test_that_multiple_observations_are_parsed_correctly(api_and_storage):
    api, storage = api_and_storage
    experiment = storage.create_experiment(
        parameters=[],
        responses=[
            SummaryConfig(
                name="summary",
                input_files=[""],
                keys=["WOPR:OP1"],
                has_finalized_keys=True,
            )
        ],
        observations={
            "summary": pl.DataFrame(
                {
                    "observation_key": [f"WOPR:OP1_o{i}" for i in range(6)],
                    "response_key": ["WOPR:OP1"] * 6,
                    "time": pl.date_range(
                        date(2000, 1, 1),
                        date(2000, 6, 1),
                        interval="1mo",
                        eager=True,
                    ),
                    "observations": [0.2, 0.1, 0.15, 0.21, 0.11, 0.151],
                    "std": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                }
            )
        },
    )
    ens = experiment.create_ensemble(ensemble_size=1, name="ensemble")

    ensemble = next(x for x in api.get_all_ensembles() if x.id == str(ens.id))
    obs_data = api.observations_for_key([ensemble.id], "WOPR:OP1")
    assert obs_data.shape == (3, 6)


def test_that_observations_for_empty_ensemble_returns_empty_data(api_and_storage):
    api, storage = api_and_storage
    experiment = storage.create_experiment(
        parameters=[],
        responses=[SummaryConfig(name="summary", input_files=[""], keys=["NAIMFRAC"])],
        observations={},
    )
    ensemble = storage.create_ensemble(experiment.id, ensemble_size=1)
    assert api.observations_for_key([str(ensemble.id)], "NAIMFRAC").empty


def test_that_data_for_response_is_empty_for_ensembles_without_responses(
    api_and_storage,
):
    api, storage = api_and_storage

    experiment = storage.create_experiment(
        parameters=[],
        responses=[
            SummaryConfig(
                name="summary",
                input_files=[],
                keys=["FOPT"],
                has_finalized_keys=True,
            )
        ],
    )

    ensemble = experiment.create_ensemble(
        ensemble_size=1, name="ensemble_without_responses"
    )

    result_df = api.data_for_response(str(ensemble.id), "FOPT")

    assert result_df.empty


def test_that_response_key_has_observation_when_only_one_experiment_has_observations(
    api_and_storage,
):
    api, storage = api_and_storage

    date = datetime(year=2024, month=10, day=4)
    experiment_with_observation = storage.create_experiment(
        parameters=[],
        responses=[
            SummaryConfig(
                name="summary",
                input_files=["CASE.UNSMRY", "CASE.SMSPEC"],
                keys=["FOPR"],
            )
        ],
        observations={
            "summary": pl.DataFrame(
                {
                    "response_key": "FOPR",
                    "observation_key": "sumobs",
                    "time": pl.Series([date]).dt.cast_time_unit("ms"),
                    "observations": pl.Series([1.0], dtype=pl.Float32),
                    "std": pl.Series([1.0], dtype=pl.Float32),
                }
            )
        },
    )

    experiment_without_observation = storage.create_experiment(
        parameters=[],
        responses=[
            SummaryConfig(
                name="summary",
                input_files=["CASE.UNSMRY", "CASE.SMSPEC"],
                keys=["FOPR"],
            )
        ],
    )

    ensemble_with_observation = experiment_with_observation.create_ensemble(
        ensemble_size=1, name="ensemble_with_obs"
    )
    ensemble_without_observation = experiment_without_observation.create_ensemble(
        ensemble_size=1, name="ensemble_without_obs"
    )

    df_summary = pl.DataFrame(
        {
            "response_key": ["FOPR"],
            "time": [pl.Series([date]).dt.cast_time_unit("ms")],
            "values": [pl.Series([1.0], dtype=pl.Float32)],
        }
    )

    ensemble_with_observation.save_response(
        "summary",
        df_summary,
        0,
    )

    ensemble_without_observation.save_response(
        "summary",
        df_summary,
        0,
    )

    with unittest.mock.patch(
        "ert.storage.Storage.experiments", new_callable=unittest.mock.PropertyMock
    ) as mock_experiments:
        mock_experiments.return_value = [
            experiment_with_observation,
            experiment_without_observation,
        ]
        responses_from_api = api.responses_api_key_defs

        assert responses_from_api[0].observations, (
            "Expected FOPR to have observations, also when only one experiment has it"
        )
