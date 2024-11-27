import contextlib
import gc
import os
import time
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Dict
from urllib.parse import quote

import httpx
import memray
import pandas as pd
import polars
import pytest
import xarray as xr
from httpx import RequestError
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


@pytest.fixture
def api_and_snake_oil_storage(snake_oil_case_storage, monkeypatch):
    with open_storage(snake_oil_case_storage.ens_path, mode="r") as storage:
        monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")
        monkeypatch.setenv("ERT_STORAGE_ENS_PATH", storage.path)

        api = PlotApi()
        yield api, storage

    if enkf._storage is not None:
        enkf._storage.close()
    enkf._storage = None
    gc.collect()


@pytest.mark.parametrize(
    "num_reals, num_dates, num_keys, max_memory_mb",
    [  # Tested 24.11.22 on macbook pro M1 max
        # (xr = tested on previous ert using xarray to store responses)
        (1, 100, 100, 1200),  # 790MiB local, xr: 791, MiB
        (1000, 100, 100, 1500),  # 809MiB local, 879MiB linux-3.11, xr: 1107MiB
        # (Cases below are more realistic at up to 200realizations)
        # Not to be run these on GHA runners
        # (2000, 100, 100, 1950),  # 1607MiB local, 1716MiB linux3.12, 1863 on linux3.11, xr: 2186MiB
        # (2, 5803, 11787, 5500),  # 4657MiB local, xr: 10115MiB
        # (10, 5803, 11787, 13500),  # 10036MiB local, 12803MiB mac-3.12, xr: 46715MiB
    ],
)
def test_plot_api_big_summary_memory_usage(
    num_reals, num_dates, num_keys, max_memory_mb, use_tmpdir, api_and_storage
):
    api, storage = api_and_storage

    dates = []

    for i in range(num_keys):
        dates += [datetime(2000, 1, 1) + timedelta(days=i)] * num_dates

    dates_df = polars.Series(dates, dtype=polars.Datetime).dt.cast_time_unit("ms")

    keys_df = polars.Series([f"K{i}" for i in range(num_keys)])
    values_df = polars.Series(list(range(num_keys * num_dates)), dtype=polars.Float32)

    big_summary = polars.DataFrame(
        {
            "response_key": polars.concat([keys_df] * num_dates),
            "time": dates_df,
            "values": values_df,
        }
    )

    experiment = storage.create_experiment(
        parameters=[],
        responses=[
            SummaryConfig(
                name="summary",
                input_files=["CASE.UNSMRY", "CASE.SMSPEC"],
                keys=keys_df,
            )
        ],
    )

    ensemble = experiment.create_ensemble(ensemble_size=num_reals, name="bigboi")
    for real in range(ensemble.ensemble_size):
        ensemble.save_response("summary", big_summary.clone(), real)

    with memray.Tracker("memray.bin", follow_fork=True, native_traces=True):
        # Initialize plotter window
        all_keys = {k.key for k in api.all_data_type_keys()}
        all_ensembles = [e.id for e in api.get_all_ensembles()]
        assert set(keys_df.to_list()) == set(all_keys)

        # call updatePlot()
        ensemble_to_data_map: Dict[str, pd.DataFrame] = {}
        sample_key = keys_df.sample(1).item()
        for ensemble in all_ensembles:
            ensemble_to_data_map[ensemble] = api.data_for_key(ensemble, sample_key)

        for ensemble in all_ensembles:
            data = ensemble_to_data_map[ensemble]

            # Transpose it twice as done in plotter
            # (should ideally be avoided)
            _ = data.T
            _ = data.T

    stats = memray._memray.compute_statistics("memray.bin")
    os.remove("memray.bin")
    total_memory_usage = stats.total_memory_allocated / (1024**2)
    assert total_memory_usage < max_memory_mb


def test_plotter_on_all_snake_oil_responses_time(api_and_snake_oil_storage):
    api, _ = api_and_snake_oil_storage
    t0 = time.time()
    key_infos = api.all_data_type_keys()
    all_ensembles = api.get_all_ensembles()
    t1 = time.time()
    # Cycle through all ensembles and get all responses
    for key_info in key_infos:
        for ensemble in all_ensembles:
            api.data_for_key(ensemble_id=ensemble.id, key=key_info.key)

        if key_info.observations:
            with contextlib.suppress(RequestError, TimeoutError):
                api.observations_for_key(
                    [ens.id for ens in all_ensembles], key_info.key
                )

        # Note: Does not test for fields
        if not (str(key_info.key).endswith("H") or "H:" in str(key_info.key)):
            with contextlib.suppress(RequestError, TimeoutError):
                api.history_data(
                    key_info.key,
                    [e.id for e in all_ensembles],
                )

    t2 = time.time()
    time_to_get_metadata = t1 - t0
    time_to_cycle_through_responses = t2 - t1

    # Local times were about 10% of the asserted times
    assert time_to_get_metadata < 1
    assert time_to_cycle_through_responses < 14


def test_plotter_on_all_snake_oil_responses_memory(api_and_snake_oil_storage):
    api, _ = api_and_snake_oil_storage

    with memray.Tracker("memray.bin", follow_fork=True, native_traces=True):
        key_infos = api.all_data_type_keys()
        all_ensembles = api.get_all_ensembles()
        # Cycle through all ensembles and get all responses
        for key_info in key_infos:
            for ensemble in all_ensembles:
                api.data_for_key(ensemble_id=ensemble.id, key=key_info.key)

            if key_info.observations:
                with contextlib.suppress(RequestError, TimeoutError):
                    api.observations_for_key(
                        [ens.id for ens in all_ensembles], key_info.key
                    )

            # Note: Does not test for fields
            if not (str(key_info.key).endswith("H") or "H:" in str(key_info.key)):
                with contextlib.suppress(RequestError, TimeoutError):
                    api.history_data(
                        key_info.key,
                        [e.id for e in all_ensembles],
                    )

    stats = memray._memray.compute_statistics("memray.bin")
    os.remove("memray.bin")
    total_memory_mb = stats.total_memory_allocated / (1024**2)
    peak_memory_mb = stats.peak_memory_allocated / (1024**2)

    # thresholds are set to about 1.5x local memory used
    assert total_memory_mb < 5000
    assert peak_memory_mb < 1500


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


def test_that_multiple_observations_are_parsed_correctly(api):
    ensemble = next(x for x in api.get_all_ensembles() if x.id == "ens_id_5")
    obs_data = api.observations_for_key([ensemble.id], "WOPR:OP1")
    assert obs_data.shape == (3, 6)
