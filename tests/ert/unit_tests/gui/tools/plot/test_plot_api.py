import gc
from datetime import datetime
from textwrap import dedent
from urllib.parse import quote

import httpx
import pandas as pd
import polars
import pytest
import xarray as xr
from pandas.testing import assert_frame_equal
from starlette.testclient import TestClient

from ert.config import GenKwConfig, SummaryConfig
from ert.dark_storage import enkf
from ert.dark_storage.app import app
from ert.gui.tools.plot.plot_api import PlotApi, PlotApiKeyDefinition
from ert.services import StorageService
from ert.storage import open_storage
from tests.ert.unit_tests.gui.tools.plot.conftest import MockResponse


@pytest.fixture(autouse=True)
def use_testclient(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(StorageService, "session", lambda: client)

    def test_escape(s: str) -> str:
        """
        Workaround for issue with TestClient:
        https://github.com/encode/starlette/issues/1060
        """
        return quote(quote(quote(s, safe="")))

    PlotApi.escape = test_escape


def test_key_def_structure(api):
    key_defs = api.all_data_type_keys()
    fopr = next(x for x in key_defs if x.key == "FOPR")
    fopr_expected = {
        "dimensionality": 2,
        "index_type": "VALUE",
        "key": "FOPR",
        "metadata": {"data_origin": "Summary"},
        "observations": True,
        "log_scale": False,
    }
    assert fopr == PlotApiKeyDefinition(**fopr_expected)

    bpr = next(x for x in key_defs if x.key == "BPR:1,3,8")
    bpr_expected = {
        "dimensionality": 2,
        "index_type": "VALUE",
        "key": "BPR:1,3,8",
        "metadata": {"data_origin": "Summary"},
        "observations": False,
        "log_scale": False,
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
    }
    assert bpr_parameter == PlotApiKeyDefinition(**bpr_parameter_expected)


def test_case_structure(api):
    ensembles = [ensemble.name for ensemble in api.get_all_ensembles()]
    hidden_case = [
        ensemble.name for ensemble in api.get_all_ensembles() if ensemble.hidden
    ]
    expected = ["ensemble_1", ".ensemble_2", "default_0", "default_1"]

    assert ensembles == expected
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
    ensemble = next(x for x in api.get_all_ensembles() if x.name == "default_0")
    for key, value in keys.items():
        observations = value["observations"]
        if observations:
            obs_data = api.observations_for_key([ensemble.id], key)
            assert not obs_data.empty
        data = api.data_for_key(ensemble.id, key)
        assert not data.empty


def test_all_data_type_keys(api):
    keys = [e.key for e in api.all_data_type_keys()]
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
    ens_id = next(ens.id for ens in api.get_all_ensembles() if ens.name == "default_0")
    df = api.history_data(ensemble_ids=[ens_id], key="FOPR")
    assert_frame_equal(
        df, pd.DataFrame({1: [0.2, 0.2, 1.2], 3: [1.0, 1.1, 1.2], 4: [1.0, 1.1, 1.3]})
    )


def test_load_history_data_searches_until_history_found(api):
    ensemble_ids = [
        ens.id
        for ens in api.get_all_ensembles()
        if ens.name in ["no-history", "default_0"]
    ]
    df = api.history_data(ensemble_ids=ensemble_ids, key="FOPR")
    assert_frame_equal(
        df, pd.DataFrame({1: [0.2, 0.2, 1.2], 3: [1.0, 1.1, 1.2], 4: [1.0, 1.1, 1.3]})
    )


def test_load_history_data_returns_empty_frame_if_no_history(api):
    ensemble_ids = [
        ens.id
        for ens in api.get_all_ensembles()
        if ens.name in ["no-history", "still-no-history"]
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
        api.all_data_type_keys()


def test_plot_api_request_errors(api):
    ensemble = next(x for x in api.get_all_ensembles() if x.name == "default_0")

    with pytest.raises(httpx.RequestError):
        api.observations_for_key([ensemble.id], "should_not_be_there")

    with pytest.raises(httpx.RequestError):
        api.data_for_key(ensemble.id, "should_not_be_there")


@pytest.fixture
def api_and_storage(monkeypatch, tmp_path):
    with open_storage(tmp_path / "storage", mode="w") as storage:
        monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")
        monkeypatch.setenv("ERT_STORAGE_ENS_PATH", storage.path)
        api = PlotApi()
        yield api, storage
    if enkf._storage is not None:
        enkf._storage.close()
    enkf._storage = None
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
            "summary": polars.DataFrame(
                {
                    "response_key": key,
                    "observation_key": "sumobs",
                    "time": polars.Series([date]).dt.cast_time_unit("ms"),
                    "observations": polars.Series([1.0], dtype=polars.Float32),
                    "std": polars.Series([1.0], dtype=polars.Float32),
                }
            )
        },
    )
    ensemble = experiment.create_ensemble(ensemble_size=1, name="ensemble")
    assert api.data_for_key(str(ensemble.id), key).empty
    df = polars.DataFrame(
        {
            "response_key": [key],
            "time": [polars.Series([date]).dt.cast_time_unit("ms")],
            "values": [polars.Series([1.0], dtype=polars.Float32)],
        }
    )
    df = df.explode("values", "time")
    ensemble.save_response(
        "summary",
        df,
        0,
    )
    assert api.data_for_key(str(ensemble.id), key).to_csv() == dedent(
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
                template_file=None,
                output_file=None,
                transform_function_definitions=[],
            ),
        ],
        responses=[],
        observations={},
    )
    ensemble = storage.create_ensemble(experiment.id, ensemble_size=10)
    assert api.data_for_key(str(ensemble.id), key).empty
    ensemble.save_parameters(
        key,
        1,
        xr.Dataset(
            {
                "values": ("names", [1.0]),
                "transformed_values": ("names", [1.0]),
                "names": [name],
            }
        ),
    )
    assert api.data_for_key(str(ensemble.id), key + ":" + name).to_csv() == dedent(
        """\
        Realization,0
        1,1.0
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
                template_file=None,
                output_file=None,
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
                "transformed_values": ("names", [1.0]),
                "names": ["key"],
            }
        ),
    )
    assert api.data_for_key(str(ensemble.id), "gen_kw").empty
    assert api.data_for_key(str(ensemble.id), "gen_kw:does_not_exist").empty
