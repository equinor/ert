import datetime
import random
import time
from dataclasses import dataclass
from textwrap import dedent

import memray
import numpy as np
import polars
import pytest

from ert.analysis import smoother_update
from ert.config import GenDataConfig, GenKwConfig, SummaryConfig
from ert.config.gen_kw_config import TransformFunctionDefinition
from ert.enkf_main import sample_prior
from ert.storage import open_storage

_rng = np.random.default_rng(0)


def _add_noise_to_df_values(df: polars.DataFrame):
    return df.with_columns(
        (
            polars.col("values")
            + polars.Series(
                "noise",
                _rng.normal(loc=0, scale=2.5, size=len(df)),
            )
        ).alias("values")
    )


@dataclass
class ExperimentInfo:
    gen_kw_config: GenKwConfig
    gen_data_config: GenDataConfig
    summary_config: SummaryConfig
    summary_observations: polars.DataFrame
    summary_responses: polars.DataFrame
    gen_data_responses: polars.DataFrame
    gen_data_observations: polars.DataFrame


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
    gen_kw_config = GenKwConfig(
        name="all_my_parameters_live_here",
        template_file=None,
        output_file=None,
        forward_init=False,
        update=True,
        transform_function_definitions=[
            TransformFunctionDefinition(
                f"param_{i}",
                param_name="NORMAL",
                values=[10, 0.1],
            )
            for i in range(num_parameters)
        ],
    )
    gen_data_config = GenDataConfig(
        name="gen_data",
        report_steps_list=[
            [list(range(num_gen_data_report_steps))] * num_gen_data_keys
        ],
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

    gen_data_responses = polars.DataFrame(
        {
            "response_key": gen_data_response_keys
            * (num_gen_data_report_steps * num_gen_data_index),
            "report_step": polars.Series(
                gendata_report_steps * (num_gen_data_keys * num_gen_data_index),
                dtype=polars.UInt16,
            ),
            "index": polars.Series(
                gendata_indexes * (num_gen_data_keys * num_gen_data_report_steps),
                dtype=polars.UInt16,
            ),
            "values": polars.Series(
                _rng.normal(loc=10, scale=0.1, size=num_gendata_rows),
                dtype=polars.Float32,
            ),
        }
    )

    gen_data_observations = polars.DataFrame(
        {
            "observation_key": gen_obs_keys,
            "response_key": random.choices(gen_data_response_keys, k=num_gen_data_obs),
            "report_step": polars.Series(
                random.choices(gendata_report_steps, k=num_gen_data_obs),
                dtype=polars.UInt16,
            ),
            "index": polars.Series(
                random.choices(gendata_indexes, k=num_gen_data_obs),
                dtype=polars.UInt16,
            ),
            "observations": polars.Series(
                _rng.normal(loc=10, scale=0.1, size=num_gen_data_obs),
                dtype=polars.Float32,
            ),
            "std": polars.Series(
                _rng.normal(loc=0.2, scale=0.1, size=num_gen_data_obs),
                dtype=polars.Float32,
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

    smry_response_key_series = polars.Series("response_key", summary_response_keys)
    smry_time_series = polars.Series(
        "time", summary_timesteps, dtype=polars.Datetime("ms")
    )
    smry_values_series = polars.Series(
        "values",
        _rng.normal(loc=10, scale=0.1, size=num_summary_keys),
        dtype=polars.Float32,
    )

    response_key_repeated = polars.concat(
        [smry_response_key_series] * num_summary_timesteps
    )
    time_repeated = polars.concat([smry_time_series] * num_summary_keys)
    values_repeated = polars.concat([smry_values_series] * num_summary_timesteps)

    # Create the DataFrame
    summary_responses = polars.DataFrame(
        {
            "response_key": response_key_repeated,
            "time": time_repeated,
            "values": values_repeated,
        }
    )

    summary_observations = polars.DataFrame(
        {
            "observation_key": [f"summary_obs_{i}" for i in range(num_summary_obs)],
            "response_key": random.choices(summary_response_keys, k=num_summary_obs),
            "time": polars.Series(
                random.choices(summary_timesteps, k=num_summary_obs),
                dtype=polars.Datetime("ms"),
            ),
            "observations": polars.Series(
                _rng.normal(loc=10, scale=0.1, size=num_summary_obs),
                dtype=polars.Float32,
            ),
            "std": polars.Series(
                _rng.normal(loc=0.2, scale=0.1, size=num_summary_obs),
                dtype=polars.Float32,
            ),
        }
    )

    return ExperimentInfo(
        gen_kw_config=gen_kw_config,
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
    time_limit_s: float
    last_measured_memory_mb: float | None = None  # For bookkeeping
    last_measured_time: float | None = None  # For bookkeeping


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
            memory_limit_mb=50,
            time_limit_s=1,
            last_measured_time=0.03,
            last_measured_memory_mb=17,
        ),
        expected_update_performance=_ExpectedPerformance(
            last_measured_time=0.09,
            last_measured_memory_mb=7.13,
            memory_limit_mb=40,
            time_limit_s=1,
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
            time_limit_s=15,
            last_measured_time=2.78,
            last_measured_memory_mb=1027,
        ),
        expected_update_performance=_ExpectedPerformance(
            memory_limit_mb=2800,
            time_limit_s=20,
            last_measured_time=5.34,
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
            time_limit_s=25,
            last_measured_memory_mb=1710,
            last_measured_time=3.59,
        ),
        expected_update_performance=_ExpectedPerformance(
            memory_limit_mb=4000,
            time_limit_s=30,
            last_measured_memory_mb=3088,
            last_measured_time=6.23,
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
            memory_limit_mb=2300,
            time_limit_s=35,
            last_measured_time=3.96,
            last_measured_memory_mb=1715,
        ),
        expected_update_performance=_ExpectedPerformance(
            memory_limit_mb=3500,
            time_limit_s=40,
            last_measured_time=6.98,
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
    #        time_limit_s=60,
    #        last_measured_time=16.03,
    #        last_measured_memory_mb=48504,
    #    ),
    #    expected_update_performance=_ExpectedPerformance(
    #        memory_limit_mb=55000,
    #        time_limit_s=60,
    #        last_measured_time=13,
    #        last_measured_memory_mb=45315,
    #    ),
    # ),
]


@pytest.mark.parametrize(
    "config, expected_performance, alias",
    [
        pytest.param(
            b.config,
            b.expected_join_performance,
            f"{b.alias}:{b.config!s}",
            id=f"{b.alias}:{b.config!s}",
        )
        for b in _BenchMarks
    ],
)
@pytest.mark.integration_test
def test_performance_of_joining_observations_and_responses(
    config: _UpdatePerfTestConfig,
    expected_performance: _ExpectedPerformance,
    alias,
    tmp_path,
):
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
            parameters=[info.gen_kw_config],
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

        with memray.Tracker(tmp_path / "memray.bin"):
            t0 = time.time()
            ens.get_observations_and_responses(
                experiment.observation_keys, np.array(range(config.num_realizations))
            )
            t1 = time.time()
            time_elapsed_s = t1 - t0

    stats = memray._memray.compute_statistics(str(tmp_path / "memray.bin"))
    mem_usage_mb = stats.total_memory_allocated / (1024**2)
    print(
        dedent(f"join,'{alias}',{round(time_elapsed_s,2)}s,{round(mem_usage_mb,2)}mb")
    )
    assert mem_usage_mb < expected_performance.memory_limit_mb
    assert time_elapsed_s < expected_performance.time_limit_s


@pytest.mark.parametrize(
    "config, expected_performance, alias",
    [
        pytest.param(
            b.config,
            b.expected_update_performance,
            f"{b.alias}:{b.config!s}",
            id=f"{b.alias}:{b.config!s}",
        )
        for b in _BenchMarks
    ],
)
@pytest.mark.integration_test
def test_performance_of_doing_es_update(
    config: _UpdatePerfTestConfig, expected_performance, alias, tmp_path
):
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
            parameters=[info.gen_kw_config],
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

        sample_prior(prior, range(config.num_realizations), [info.gen_kw_config.name])
        posterior = experiment.create_ensemble(
            ensemble_size=config.num_realizations,
            name="AtleJohnny",
            prior_ensemble=prior,
            iteration=1,
        )
        with memray.Tracker(tmp_path / "memray.bin"):
            t0 = time.time()
            smoother_update(
                prior,
                posterior,
                prior.experiment.observation_keys,
                [info.gen_kw_config.name],
            )
            t1 = time.time()
            time_elapsed_s = t1 - t0

    stats = memray._memray.compute_statistics(str(tmp_path / "memray.bin"))
    mem_usage_mb = stats.total_memory_allocated / (1024**2)
    print(
        dedent(f"update,'{alias}',{round(time_elapsed_s,2)}s,{round(mem_usage_mb,2)}mb")
    )
    assert mem_usage_mb < expected_performance.memory_limit_mb
    assert time_elapsed_s < expected_performance.time_limit_s
