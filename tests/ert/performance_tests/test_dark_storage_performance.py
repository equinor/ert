import contextlib
import gc
import io
import os
import time
from asyncio import get_event_loop
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TypeVar
from urllib.parse import quote

import memray
import numpy as np
import pandas as pd
import pytest
from httpx import RequestError
from starlette.testclient import TestClient

from ert.config import ErtConfig
from ert.dark_storage import enkf
from ert.dark_storage.app import app
from ert.dark_storage.endpoints import ensembles, experiments, records
from ert.gui.tools.plot.plot_api import PlotApi
from ert.libres_facade import LibresFacade
from ert.services import StorageService
from ert.storage import open_storage
from tests.ert.performance_tests.test_obs_and_responses_performance import (
    ExpectedPerformance,
    StoragePerfTestConfig,
    create_experiment_args,
)

T = TypeVar("T")


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


def run_in_loop(coro: Awaitable[T]) -> T:
    return get_event_loop().run_until_complete(coro)


def get_single_record_csv(storage, ensemble_id1, keyword, poly_ran):
    csv = run_in_loop(
        records.get_ensemble_record(
            storage=storage,
            name=keyword,
            ensemble_id=ensemble_id1,
        )
    ).body
    record_df1_indexed = pd.read_csv(
        io.BytesIO(csv), index_col=0, float_precision="round_trip"
    )
    assert len(record_df1_indexed.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1_indexed.index) == 1


def get_record_observations(storage, ensemble_id, keyword: str, poly_ran):
    obs = run_in_loop(
        records.get_record_observations(
            storage=storage, ensemble_id=ensemble_id, response_name=keyword
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
            assert np.isclose(obs[0].values[0], 2.6357)
            assert len(obs[0].errors) == 1
            assert len(obs[0].x_axis) == 1
            assert len(obs[0].values) == 1
        else:
            assert len(obs) == 0

    elif "POLY_RES_" in keyword:
        n = int(keyword.split("@")[0][9:])
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
        records.get_ensemble_record(
            storage=storage,
            name=keyword,
            ensemble_id=ensemble_id1,
            accept="application/x-parquet",
        )
    ).body
    record_df1 = pd.read_parquet(io.BytesIO(parquet))
    assert len(record_df1.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1.index) == poly_ran["reals"]


def get_record_csv(storage, ensemble_id1, keyword, poly_ran):
    csv = run_in_loop(
        records.get_ensemble_record(
            storage=storage, name=keyword, ensemble_id=ensemble_id1
        )
    ).body
    record_df1 = pd.read_csv(io.BytesIO(csv), index_col=0, float_precision="round_trip")
    assert len(record_df1.columns) == poly_ran["gen_data_entries"]
    assert len(record_df1.index) == poly_ran["reals"]


def get_parameters(storage, ensemble_id1, keyword, poly_ran):
    parameters_json = run_in_loop(
        records.get_ensemble_parameters(storage=storage, ensemble_id=ensemble_id1)
    )
    assert (
        len(parameters_json)
        == poly_ran["parameter_entries"] * poly_ran["parameter_count"]
    )


@pytest.mark.parametrize(
    "function",
    [
        get_record_parquet,
        get_record_csv,
        get_parameters,
    ],
)
@pytest.mark.parametrize(
    "keyword", ["summary", "gen_data", "summary_with_obs", "gen_data_with_obs"]
)
@pytest.mark.integration_test
def test_direct_dark_performance(
    benchmark, template_config, monkeypatch, function, keyword
):
    key = {
        "summary": "PSUM1",
        "gen_data": "POLY_RES_1@0",
        "summary_with_obs": "PSUM0",
        "gen_data_with_obs": "POLY_RES_0@0",
    }[keyword]

    with template_config["folder"].as_cwd():
        config = ErtConfig.from_file("poly.ert")
        enkf_facade = LibresFacade(config)
        storage = open_storage(enkf_facade.enspath)
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
@pytest.mark.integration_test
def test_direct_dark_performance_with_storage(
    benchmark, template_config, monkeypatch, function, keyword
):
    key = {
        "summary": "PSUM1",
        "gen_data": "POLY_RES_1@0",
        "summary_with_obs": "PSUM0",
        "gen_data_with_obs": "POLY_RES_0@0",
    }[keyword]

    with template_config["folder"].as_cwd():
        config = ErtConfig.from_file("poly.ert")
        enkf_facade = LibresFacade(config)
        storage = open_storage(enkf_facade.enspath)
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


@dataclass
class _OpenPlotterBenchmark:
    alias: str
    config: StoragePerfTestConfig
    expected_performance: ExpectedPerformance


open_plotter_benchmarks: list[_OpenPlotterBenchmark] = [
    _OpenPlotterBenchmark(
        alias="small",
        config=StoragePerfTestConfig(
            num_parameters=1,
            num_gen_data_keys=100,
            num_gen_data_report_steps=2,
            num_gen_data_index=10,
            num_gen_data_obs=400,
            num_summary_keys=1,
            num_summary_timesteps=1,
            num_summary_obs=1,
            num_realizations=2,
        ),
        expected_performance=ExpectedPerformance(memory_limit_mb=1e9, time_limit_s=1e9),
    ),
    _OpenPlotterBenchmark(
        alias="medium",
        config=StoragePerfTestConfig(
            num_parameters=1,
            num_gen_data_keys=2000,
            num_gen_data_report_steps=2,
            num_gen_data_index=10,
            num_gen_data_obs=2000,
            num_summary_keys=1000,
            num_summary_timesteps=200,
            num_summary_obs=2000,
            num_realizations=200,
        ),
        expected_performance=ExpectedPerformance(memory_limit_mb=1e9, time_limit_s=1e9),
    ),
    _OpenPlotterBenchmark(
        alias="large",
        config=StoragePerfTestConfig(
            num_parameters=1,
            num_gen_data_keys=5000,
            num_gen_data_report_steps=2,
            num_gen_data_index=10,
            num_gen_data_obs=5000,
            num_summary_keys=1000,
            num_summary_timesteps=200,
            num_summary_obs=2000,
            num_realizations=200,
        ),
        expected_performance=ExpectedPerformance(memory_limit_mb=1e9, time_limit_s=1e9),
    ),
    _OpenPlotterBenchmark(
        alias="large+",
        config=StoragePerfTestConfig(
            num_parameters=1,
            num_gen_data_keys=5000,
            num_gen_data_report_steps=2,
            num_gen_data_index=10,
            num_gen_data_obs=5000,
            num_summary_keys=1000,
            num_summary_timesteps=200,
            num_summary_obs=5000,
            num_realizations=200,
        ),
        expected_performance=ExpectedPerformance(memory_limit_mb=1e9, time_limit_s=1e9),
    ),
    _OpenPlotterBenchmark(
        alias="large++",
        config=StoragePerfTestConfig(
            num_parameters=5000,
            num_gen_data_keys=5000,
            num_gen_data_report_steps=2,
            num_gen_data_index=10,
            num_gen_data_obs=5000,
            num_summary_keys=17000,
            num_summary_timesteps=200,
            num_summary_obs=5000,
            num_realizations=200,
        ),
        expected_performance=ExpectedPerformance(memory_limit_mb=1e9, time_limit_s=1e9),
    ),
]


@pytest.mark.parametrize("plotter_benchmark", open_plotter_benchmarks)
def test_plot_api_big_summary_memory_usage(
    plotter_benchmark, use_tmpdir, api_and_storage
):
    api, storage = api_and_storage
    config = plotter_benchmark.config
    expected_performance = plotter_benchmark.expected_performance

    info = create_experiment_args(
        num_parameters=config.num_parameters,
        num_gen_data_keys=config.num_gen_data_keys,
        num_gen_data_report_steps=config.num_gen_data_report_steps,
        num_gen_data_index=config.num_gen_data_index,
        num_gen_data_obs=config.num_gen_data_obs,
        num_summary_keys=config.num_summary_keys,
        num_summary_timesteps=config.num_summary_timesteps,
        num_summary_obs=config.num_summary_obs,
    )

    experiment = storage.create_experiment(
        parameters=[info.gen_kw_config],
        responses=[info.summary_config, info.gen_data_config],
        observations={
            "gen_data": info.gen_data_observations,
            "summary": info.summary_observations,
        },
    )

    ens = experiment.create_ensemble(
        ensemble_size=config.num_realizations, name="bigboi"
    )

    for real in range(config.num_realizations):
        ens.save_response("summary", info.summary_responses.clone(), real)
        ens.save_response("gen_data", info.gen_data_responses.clone(), real)

    with memray.Tracker("memray.bin", follow_fork=True, native_traces=True):
        # Initialize plotter window
        t0 = time.time()
        _ = {k.key for k in api.all_data_type_keys()}
        all_ensembles = [e.id for e in api.get_all_ensembles()]

        # call updatePlot()
        ensemble_to_data_map: dict[str, pd.DataFrame] = {}
        sample_key = info.summary_responses["response_key"].sample(1).item()
        for ensemble in all_ensembles:
            ensemble_to_data_map[ensemble] = api.data_for_key(ensemble, sample_key)

        for ensemble in all_ensembles:
            data = ensemble_to_data_map[ensemble]

            # Transpose it twice as done in plotter
            # (should ideally be avoided)
            _ = data.T
            _ = data.T

        time_elapsed = time.time() - t0

    stats = memray._memray.compute_statistics("memray.bin")
    os.remove("memray.bin")
    total_memory_usage = stats.total_memory_allocated / (1024**2)
    assert total_memory_usage < expected_performance.memory_limit_mb
    assert time_elapsed < expected_performance.time_limit_s
    print(
        f"{plotter_benchmark.alias}[{plotter_benchmark.config}] took {total_memory_usage}mb,{time_elapsed}s"
    )


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
