import datetime
import random
from dataclasses import dataclass

import memray
import numpy as np
import polars as pl
import pytest

from ert.analysis import enif_update, smoother_update
from ert.config import (
    ESSettings,
    GenDataConfig,
    GenKwConfig,
    ObservationSettings,
    SummaryConfig,
)
from ert.sample_prior import sample_prior
from ert.storage import open_storage

_rng = np.random.default_rng(0)


def _add_noise_to_df_values(df: pl.DataFrame):
    return df.with_columns(
        (
            pl.col("values")
            + pl.Series(
                "noise",
                _rng.normal(loc=0, scale=2.5, size=len(df)),
            )
        ).alias("values")
    )


@dataclass
class ExperimentInfo:
    gen_kw_configs: list[GenKwConfig]
    gen_data_config: GenDataConfig
    summary_config: SummaryConfig
    summary_observations: pl.DataFrame
    summary_responses: pl.DataFrame
    gen_data_responses: pl.DataFrame
    gen_data_observations: pl.DataFrame


def create_experiment_args(
    num_parameters: int,
    num_gen_data_keys: int,
    num_gen_data_report_steps: int,
    num_gen_data_index: int,
    num_gen_data_obs: int,
    num_summary_keys: int,
    num_summary_timesteps: int,
    num_summary_obs: int,
) -> ExperimentInfo:
    gen_kw_configs = [
        GenKwConfig(
            name=f"param_{i}",
            group="all_my_parameters_live_here",
            distribution={"name": "normal", "mean": 10, "std": 0.1},
        )
        for i in range(num_parameters)
    ]
    gen_data_config = GenDataConfig(
        name="gen_data",
        report_steps_list=[list(range(num_gen_data_report_steps))] * num_gen_data_keys,
        keys=[f"gen_data_{i}" for i in range(num_gen_data_keys)],
    )

    # Remember to do one explicit .save_parameters to an ensemble
    # to get the finalized summary keys stored in the experiment
    summary_config = SummaryConfig(name="summary", keys=["*"])

    # Now create the observations
    gen_obs_keys = [f"genobs_{i}" for i in range(num_gen_data_obs)]
    gendata_indexes = [random.randint(0, 10000) for _ in range(num_gen_data_index)]
    gendata_report_steps = [
        random.randint(0, 500) for _ in range(num_gen_data_report_steps)
    ]
    num_gendata_rows = (
        num_gen_data_keys * num_gen_data_report_steps * num_gen_data_index
    )

    gen_data_response_keys = [f"gen_data_{i}" for i in range(num_gen_data_keys)]

    gen_data_responses = pl.DataFrame(
        {
            "response_key": gen_data_response_keys
            * (num_gen_data_report_steps * num_gen_data_index),
            "report_step": pl.Series(
                gendata_report_steps * (num_gen_data_keys * num_gen_data_index),
                dtype=pl.UInt16,
            ),
            "index": pl.Series(
                gendata_indexes * (num_gen_data_keys * num_gen_data_report_steps),
                dtype=pl.UInt16,
            ),
            "values": pl.Series(
                _rng.normal(loc=10, scale=0.1, size=num_gendata_rows),
                dtype=pl.Float32,
            ),
        }
    )

    gen_data_observations = pl.DataFrame(
        {
            "observation_key": gen_obs_keys,
            "response_key": random.choices(gen_data_response_keys, k=num_gen_data_obs),
            "report_step": pl.Series(
                random.choices(gendata_report_steps, k=num_gen_data_obs),
                dtype=pl.UInt16,
            ),
            "index": pl.Series(
                random.choices(gendata_indexes, k=num_gen_data_obs),
                dtype=pl.UInt16,
            ),
            "observations": pl.Series(
                _rng.normal(loc=10, scale=0.1, size=num_gen_data_obs),
                dtype=pl.Float32,
            ),
            "std": pl.Series(
                _rng.normal(loc=0.2, scale=0.1, size=num_gen_data_obs),
                dtype=pl.Float32,
            ),
        }
    )

    _summary_time_delta = datetime.timedelta(30)
    summary_timesteps = [
        datetime.datetime(2000, 1, 1) + datetime.timedelta(30) * i
        for i in range(num_summary_timesteps)
    ]
    summary_response_keys = [
        # Note: Must correspond to response_key in generated summary response ds
        f"summary_{i}"
        for i in range(num_summary_keys)
    ]

    smry_response_key_series = pl.Series("response_key", summary_response_keys)
    smry_time_series = pl.Series("time", summary_timesteps, dtype=pl.Datetime("ms"))
    smry_values_series = pl.Series(
        "values",
        _rng.normal(loc=10, scale=0.1, size=num_summary_keys),
        dtype=pl.Float32,
    )

    response_key_repeated = pl.concat(
        [smry_response_key_series] * num_summary_timesteps
    )
    time_repeated = pl.concat([smry_time_series] * num_summary_keys)
    values_repeated = pl.concat([smry_values_series] * num_summary_timesteps)

    # Create the DataFrame
    summary_responses = pl.DataFrame(
        {
            "response_key": response_key_repeated,
            "time": time_repeated,
            "values": values_repeated,
        }
    )

    summary_observations = pl.DataFrame(
        {
            "observation_key": [f"summary_obs_{i}" for i in range(num_summary_obs)],
            "response_key": random.choices(summary_response_keys, k=num_summary_obs),
            "time": pl.Series(
                random.choices(summary_timesteps, k=num_summary_obs),
                dtype=pl.Datetime("ms"),
            ),
            "observations": pl.Series(
                _rng.normal(loc=10, scale=0.1, size=num_summary_obs),
                dtype=pl.Float32,
            ),
            "std": pl.Series(
                _rng.normal(loc=0.2, scale=0.1, size=num_summary_obs),
                dtype=pl.Float32,
            ),
        }
    )

    return ExperimentInfo(
        gen_kw_configs=gen_kw_configs,
        gen_data_config=gen_data_config,
        summary_config=summary_config,
        summary_observations=summary_observations,
        summary_responses=summary_responses,
        gen_data_responses=gen_data_responses,
        gen_data_observations=gen_data_observations,
    )


@dataclass
class _UpdatePerfTestConfig:
    num_parameters: int
    num_gen_data_keys: int
    num_gen_data_report_steps: int
    num_gen_data_index: int
    num_gen_data_obs: int
    num_summary_keys: int
    num_summary_timesteps: int
    num_summary_obs: int
    num_realizations: int

    def __str__(self) -> str:
        gen_data_rows = (
            self.num_gen_data_keys
            * self.num_gen_data_index
            * self.num_gen_data_report_steps
        )
        smry_rows = self.num_summary_keys * self.num_summary_timesteps
        return (
            f"[{self.num_realizations}rls|"
            f"{self.num_parameters}p|"
            f"gd:{self.num_gen_data_obs}o->{gen_data_rows}r|"
            f"smr:{self.num_summary_obs}o->{smry_rows}r]"
        )


@dataclass
class _ExpectedPerformance:
    memory_limit_mb: float
    last_measured_memory_mb: float | None = None  # For bookkeeping


@dataclass
class _Benchmark:
    alias: str
    config: _UpdatePerfTestConfig
    expected_update_performance: _ExpectedPerformance
    expected_join_performance: _ExpectedPerformance


# We want to apply these benchmarks
# the same for update and for join in-separate
# hence they are all declared here
# Note: Adjusting num responses did not seem
# to have a very big impact on performance.
_BenchMarks: list[_Benchmark] = [
    _Benchmark(
        alias="small",
        config=_UpdatePerfTestConfig(
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
        expected_join_performance=_ExpectedPerformance(
            memory_limit_mb=100,
            last_measured_memory_mb=17,
        ),
        expected_update_performance=_ExpectedPerformance(
            last_measured_memory_mb=7.13,
            memory_limit_mb=100,
        ),
    ),
    _Benchmark(
        alias="medium",
        config=_UpdatePerfTestConfig(
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
        expected_join_performance=_ExpectedPerformance(
            memory_limit_mb=1500,
            last_measured_memory_mb=1027,
        ),
        expected_update_performance=_ExpectedPerformance(
            memory_limit_mb=3100,
            last_measured_memory_mb=2230,
        ),
    ),
    _Benchmark(
        alias="large",
        config=_UpdatePerfTestConfig(
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
        expected_join_performance=_ExpectedPerformance(
            memory_limit_mb=4500,
            last_measured_memory_mb=1710,
        ),
        expected_update_performance=_ExpectedPerformance(
            memory_limit_mb=4000,
            last_measured_memory_mb=3088,
        ),
    ),
    _Benchmark(
        alias="large+",
        config=_UpdatePerfTestConfig(
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
        expected_join_performance=_ExpectedPerformance(
            memory_limit_mb=3300,
            last_measured_memory_mb=1715,
        ),
        expected_update_performance=_ExpectedPerformance(
            memory_limit_mb=4500,
            last_measured_memory_mb=3115,
        ),
    ),
    # Only run locally
    # _Benchmark(
    #    alias="bigrealcase",
    #    config=_UpdatePerfTestConfig(
    #        num_parameters=5000,
    #        num_gen_data_keys=5000,
    #        num_gen_data_report_steps=2,
    #        num_gen_data_index=10,
    #        num_gen_data_obs=5000,
    #        num_summary_keys=24868,
    #        num_summary_timesteps=6000,
    #        num_summary_obs=17000,
    #        num_realizations=2,
    #    ),
    #    expected_join_performance=_ExpectedPerformance(
    #        memory_limit_mb=55000,
    #        last_measured_memory_mb=48504,
    #    ),
    #    expected_update_performance=_ExpectedPerformance(
    #        memory_limit_mb=55000,
    #        last_measured_memory_mb=45315,
    #    ),
    # ),
]


@pytest.fixture(
    params=[
        (
            b.alias,
            b.config,
            b.expected_join_performance,
        )
        for b in _BenchMarks
    ],
)
def setup_benchmark(tmp_path, request):
    alias, config, expected_performance = request.param
    info = create_experiment_args(
        config.num_parameters,
        config.num_gen_data_keys,
        config.num_gen_data_report_steps,
        config.num_gen_data_index,
        config.num_gen_data_obs,
        config.num_summary_keys,
        config.num_summary_timesteps,
        config.num_summary_obs,
    )

    with open_storage(tmp_path / "storage", mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[info.gen_data_config, info.summary_config],
            parameters=info.gen_kw_configs,
            observations={
                "gen_data": info.gen_data_observations,
                "summary": info.summary_observations,
            },
        )
        ens = experiment.create_ensemble(
            ensemble_size=config.num_realizations, name="BobKaareJohnny"
        )

        for real in range(config.num_realizations):
            ens.save_response("summary", info.summary_responses.clone(), real)
            ens.save_response("gen_data", info.gen_data_responses.clone(), real)

        yield (
            alias,
            ens,
            experiment.observation_keys,
            np.array(range(config.num_realizations)),
            expected_performance,
        )


@pytest.mark.memory_test
def test_memory_performance_of_joining_observations_and_responses(
    setup_benchmark, tmp_path
):
    _, ens, observation_keys, mask, expected_performance = setup_benchmark

    with memray.Tracker(tmp_path / "memray.bin"):
        ens.get_observations_and_responses(observation_keys, mask)

    stats = memray._memray.compute_statistics(str(tmp_path / "memray.bin"))
    mem_usage_mb = stats.total_memory_allocated / (1024**2)
    assert mem_usage_mb < expected_performance.memory_limit_mb


def test_time_performance_of_joining_observations_and_responses(
    setup_benchmark, benchmark
):
    alias, ens, observation_keys, mask, _ = setup_benchmark

    if alias != "small":
        pytest.skip()

    def run():
        ens.get_observations_and_responses(observation_keys, mask)

    benchmark(run)


@pytest.fixture(
    params=[
        (
            b.alias,
            b.config,
            b.expected_update_performance,
        )
        for b in _BenchMarks
    ],
)
def setup_es_benchmark(tmp_path, request):
    alias, config, expected_performance = request.param
    info = create_experiment_args(
        config.num_parameters,
        config.num_gen_data_keys,
        config.num_gen_data_report_steps,
        config.num_gen_data_index,
        config.num_gen_data_obs,
        config.num_summary_keys,
        config.num_summary_timesteps,
        config.num_summary_obs,
    )

    with open_storage(tmp_path / "storage", mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[info.gen_data_config, info.summary_config],
            parameters=info.gen_kw_configs,
            observations={
                "gen_data": info.gen_data_observations,
                "summary": info.summary_observations,
            },
        )
        prior = experiment.create_ensemble(
            ensemble_size=config.num_realizations, name="BobKaareJohnny", iteration=0
        )

        for real in range(config.num_realizations):
            # Note: We add noise to avoid ensemble collapse in which
            # case the update won't happen
            prior.save_response(
                "summary",
                _add_noise_to_df_values(info.summary_responses),
                real,
            )
            prior.save_response(
                "gen_data", _add_noise_to_df_values(info.gen_data_responses), real
            )

        sample_prior(
            prior,
            range(config.num_realizations),
            42,
            [c.name for c in info.gen_kw_configs],
        )
        posterior = experiment.create_ensemble(
            ensemble_size=config.num_realizations,
            name="AtleJohnny",
            prior_ensemble=prior,
            iteration=1,
        )

        yield (
            alias,
            prior,
            posterior,
            [cfg.name for cfg in info.gen_kw_configs],
            expected_performance,
        )


@pytest.mark.memory_test
def test_memory_performance_of_doing_es_update(setup_es_benchmark, tmp_path):
    _, prior, posterior, gen_kw_names, expected_performance = setup_es_benchmark
    with memray.Tracker(tmp_path / "memray.bin"):
        smoother_update(
            prior,
            posterior,
            prior.experiment.observation_keys,
            gen_kw_names,
            ObservationSettings(),
            ESSettings(),
        )

    stats = memray._memray.compute_statistics(str(tmp_path / "memray.bin"))
    mem_usage_mb = stats.total_memory_allocated / (1024**2)
    assert mem_usage_mb < expected_performance.memory_limit_mb


def test_speed_performance_of_doing_es_update(setup_es_benchmark, benchmark):
    alias, prior, posterior, gen_kw_names, _ = setup_es_benchmark

    if alias != "small":
        pytest.skip()

    def run():
        smoother_update(
            prior,
            posterior,
            prior.experiment.observation_keys,
            gen_kw_names,
            ObservationSettings(),
            ESSettings(),
        )

    benchmark(run)


@pytest.mark.memory_test
def test_memory_performance_of_doing_enif_update(setup_es_benchmark, tmp_path):
    _, prior, posterior, gen_kw_names, expected_performance = setup_es_benchmark
    with memray.Tracker(tmp_path / "memray.bin"):
        enif_update(
            prior,
            posterior,
            prior.experiment.observation_keys,
            gen_kw_names,
            12345,
        )

    stats = memray._memray.compute_statistics(str(tmp_path / "memray.bin"))
    mem_usage_mb = stats.total_memory_allocated / (1024**2)
    assert mem_usage_mb < expected_performance.memory_limit_mb


def test_speed_performance_of_doing_enif_update(setup_es_benchmark, benchmark):
    alias, prior, posterior, gen_kw_names, _ = setup_es_benchmark

    if alias != "small":
        pytest.skip()

    def run():
        enif_update(
            prior,
            posterior,
            prior.experiment.observation_keys,
            gen_kw_names,
            123456789,
        )

    benchmark(run)
