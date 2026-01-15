import asyncio
import contextlib
import gc
import io
import json
import os
from collections.abc import Awaitable
from datetime import datetime, timedelta
from typing import TypeVar
from urllib.parse import quote
from uuid import UUID

import memray
import numpy as np
import pandas as pd
import polars as pl
import pytest
from httpx import RequestError
from starlette.testclient import TestClient

from ert.config import SummaryConfig
from ert.dark_storage import common
from ert.dark_storage.app import app
from ert.dark_storage.endpoints import ensembles, experiments
from ert.dark_storage.endpoints.observations import get_observations_for_response
from ert.dark_storage.endpoints.responses import get_response
from ert.gui.tools.plot import plot_api
from ert.gui.tools.plot.plot_api import PlotApi
from ert.storage import Storage, open_storage

T = TypeVar("T")


def get_response_autofilter(
    storage: Storage, response_key: str, ensemble_id: UUID, **kwargs
):
    if "@" in response_key:
        response_key, report_step = response_key.split("@")
        return get_response(
            storage=storage,
            response_key=response_key,
            ensemble_id=ensemble_id,
            filter_on=json.dumps({"report_step": report_step}),
            **kwargs,
        )

    return get_response(
        storage=storage,
        ensemble_id=ensemble_id,
        response_key=response_key,
        filter_on=None,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def use_testclient(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(plot_api, "create_ertserver_client", lambda project: client)

    def test_escape(s: str) -> str:
        """
        Workaround for issue with TestClient:
        https://github.com/encode/starlette/issues/1060
        """
        return quote(quote(quote(s, safe="")))

    PlotApi.escape = test_escape


def run_in_loop(coro: Awaitable[T]) -> T:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def get_record_observations(storage, ensemble_id, keyword: str, poly_ran):
    response_key, report_step = (
        keyword.split("@") if "@" in keyword else (keyword, None)
    )

    obs = run_in_loop(
        get_observations_for_response(
            storage=storage,
            ensemble_id=ensemble_id,
            response_key=response_key,
            filter_on=json.dumps({"report_step": report_step})
            if report_step is not None
            else None,
        )
    )

    if "PSUM" in keyword:
        n = int(keyword[4:])
        if n < poly_ran["sum_obs_count"]:
            num_summary_obs = poly_ran["sum_obs_count"] * (
                poly_ran["summary_data_entries"] // poly_ran["sum_obs_every"]
            )
            assert len(obs) == num_summary_obs
            assert np.isclose(obs[0].errors[0], 0.1)
            assert obs[0].x_axis[0].startswith("2010-01-02T00:00:00")
            assert np.isclose(obs[0].values[0], 2.6357)  # noqa: PD011
            assert len(obs[0].errors) == 1
            assert len(obs[0].x_axis) == 1
            assert len(obs[0].values) == 1
        else:
            assert len(obs) == 0

    elif "POLY_RES_" in keyword:
        n = int(keyword.split("@", maxsplit=1)[0][9:])
        if n < poly_ran["gen_obs_count"]:
            num_general_obs = poly_ran["gen_obs_count"]
            rows_per_obs = poly_ran["gen_data_entries"] // poly_ran["gen_obs_every"]
            assert len(obs) == num_general_obs
            assert len(obs[0].errors) == rows_per_obs
            assert len(obs[0].x_axis) == rows_per_obs
            assert len(obs[0].values) == rows_per_obs
        else:
            assert len(obs) == 0
    else:
        raise AssertionError(f"should never get here, keyword is {keyword}")


def get_record_parquet(storage, ensemble_id1, keyword, poly_ran):
    parquet = run_in_loop(
        get_response_autofilter(
            storage=storage,
            response_key=keyword,
            ensemble_id=ensemble_id1,
            accept="application/x-parquet",
        )
    ).body
    record_df1 = pd.read_parquet(io.BytesIO(parquet))
    assert len(record_df1.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1.index) == poly_ran["reals"]


def get_record_csv(storage, ensemble_id1, keyword, poly_ran):
    csv = run_in_loop(
        get_response_autofilter(
            storage=storage, response_key=keyword, ensemble_id=ensemble_id1
        )
    ).body
    record_df1 = pd.read_csv(io.BytesIO(csv), index_col=0, float_precision="round_trip")
    assert len(record_df1.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1.index) == poly_ran["reals"]


@pytest.mark.parametrize(
    "function",
    [get_record_parquet, get_record_csv],
)
@pytest.mark.parametrize(
    "keyword", ["summary", "gen_data", "summary_with_obs", "gen_data_with_obs"]
)
@pytest.mark.filterwarnings(
    "ignore:Config contains a SUMMARY key but no forward model steps"
)
def test_direct_dark_performance(benchmark, template_config, function, keyword):
    key = {
        "summary": "PSUM1",
        "gen_data": "POLY_RES_1@0",
        "summary_with_obs": "PSUM0",
        "gen_data_with_obs": "POLY_RES_0@0",
    }[keyword]

    with template_config["folder"].as_cwd():
        storage = open_storage("storage")
        experiment_json = experiments.get_experiments(storage=storage)
        ensemble_id_default = None
        for ensemble_id in experiment_json[0].ensemble_ids:
            ensemble_json = ensembles.get_ensemble(
                storage=storage, ensemble_id=ensemble_id
            )
            if ensemble_json.userdata["name"] == "default":
                ensemble_id_default = ensemble_id

        benchmark(function, storage, ensemble_id_default, key, template_config)


@pytest.mark.parametrize(
    "function",
    [
        get_record_observations,
    ],
)
@pytest.mark.parametrize(
    "keyword", ["summary", "gen_data", "summary_with_obs", "gen_data_with_obs"]
)
def test_direct_dark_performance_with_storage(
    benchmark, template_config, function, keyword
):
    key = {
        "summary": "PSUM1",
        "gen_data": "POLY_RES_1@0",
        "summary_with_obs": "PSUM0",
        "gen_data_with_obs": "POLY_RES_0@0",
    }[keyword]

    with template_config["folder"].as_cwd():
        storage = open_storage("storage")
        experiment_json = experiments.get_experiments(storage=storage)
        ensemble_id_default = None
        for ensemble_id in experiment_json[0].ensemble_ids:
            ensemble_json = ensembles.get_ensemble(
                storage=storage, ensemble_id=ensemble_id
            )
            if ensemble_json.userdata["name"] == "default":
                ensemble_id_default = ensemble_id

        benchmark(function, storage, ensemble_id_default, key, template_config)


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


@pytest.fixture
def api_and_snake_oil_storage(snake_oil_case_storage, monkeypatch):
    with open_storage(snake_oil_case_storage.ens_path, mode="r") as storage:
        monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")
        monkeypatch.setenv("ERT_STORAGE_ENS_PATH", str(storage.path))

        api = PlotApi(snake_oil_case_storage.ens_path)
        yield api, storage

    if common._storage is not None:
        common._storage.close()
    common._storage = None
    gc.collect()


@pytest.mark.memory_test
@pytest.mark.parametrize(
    ("num_reals", "num_dates", "num_keys", "max_memory_mb"),
    [  # Tested 24.11.22 on macbook pro M1 max
        # (xr = tested on previous ert using xarray to store responses)
        (1, 100, 100, 2000),  # 790MiB local, xr: 791, MiB
        (1000, 100, 100, 3000),  # 809MiB local, 879MiB linux-3.11, xr: 1107MiB
        # (Cases below are more realistic at up to 200realizations)
        # Not to be run these on GHA runners
        # (2000, 100, 100, 1950),
        #   1607MiB local, 1716MiB linux3.12, 1863 on linux3.11, xr: 2186MiB
        # (2, 5803, 11787, 5500),  # 4657MiB local, xr: 10115MiB
        # (10, 5803, 11787, 13500),  # 10036MiB local, 12803MiB mac-3.12, xr: 46715MiB
    ],
)
@pytest.mark.flaky(reruns=2)
def test_plot_api_big_summary_memory_usage(
    num_reals, num_dates, num_keys, max_memory_mb, use_tmpdir, api_and_storage
):
    api, storage = api_and_storage

    dates = []

    for i in range(num_keys):
        dates += [datetime(2000, 1, 1) + timedelta(days=i)] * num_dates

    dates_df = pl.Series(dates, dtype=pl.Datetime).dt.cast_time_unit("ms")

    keys_df = pl.Series([f"K{i}" for i in range(num_keys)])
    values_df = pl.Series(list(range(num_keys * num_dates)), dtype=pl.Float32)

    big_summary = pl.DataFrame(
        {
            "response_key": pl.concat([keys_df] * num_dates),
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
        response_keys = {k.key for k in api.responses_api_key_defs}
        all_ensembles = [e.id for e in api.get_all_ensembles()]
        assert set(keys_df.to_list()) == set(response_keys)

        # call updatePlot()
        ensemble_to_data_map: dict[str, pd.DataFrame] = {}
        sample_key = keys_df.sample(1).item()
        for ensemble in all_ensembles:
            ensemble_to_data_map[ensemble] = api.data_for_response(ensemble, sample_key)

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


def test_plotter_on_all_snake_oil_responses_time(api_and_snake_oil_storage, benchmark):
    api, _ = api_and_snake_oil_storage

    def run():
        key_infos_params = api.parameters_api_key_defs
        key_infos_responses = api.responses_api_key_defs
        all_ensembles = api.get_all_ensembles()
        # Cycle through all ensembles and get all responses
        for key_info in key_infos_params:
            for ensemble in all_ensembles:
                api.data_for_parameter(
                    ensemble_id=ensemble.id, parameter_key=key_info.parameter.name
                )

        for key_info in key_infos_responses:
            for ensemble in all_ensembles:
                api.data_for_response(
                    ensemble_id=ensemble.id,
                    response_key=key_info.key,
                    filter_on=key_info.filter_on,
                )

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
                        filter_on=key_info.filter_on,
                    )

    benchmark(run)


@pytest.mark.memory_test
def test_plotter_on_all_snake_oil_responses_memory(api_and_snake_oil_storage):
    api, _ = api_and_snake_oil_storage

    with memray.Tracker("memray.bin", follow_fork=True, native_traces=True):
        key_infos = api.responses_api_key_defs
        all_ensembles = api.get_all_ensembles()
        # Cycle through all ensembles and get all responses
        for key_info in key_infos:
            for ensemble in all_ensembles:
                api.data_for_response(
                    ensemble_id=ensemble.id,
                    response_key=key_info.key,
                    filter_on=key_info.filter_on,
                )

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
                        filter_on=key_info.filter_on,
                    )

    stats = memray._memray.compute_statistics("memray.bin")
    os.remove("memray.bin")
    total_memory_mb = stats.total_memory_allocated / (1024**2)
    peak_memory_mb = stats.peak_memory_allocated / (1024**2)

    # thresholds are set to about 1.5x local memory used
    assert total_memory_mb < 5000
    assert peak_memory_mb < 1500
