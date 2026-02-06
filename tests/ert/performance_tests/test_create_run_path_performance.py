import itertools

import numpy as np
import polars as pl
import pytest

from ert.config import GenKwConfig
from ert.run_arg import create_run_arguments
from ert.run_models._create_run_path import create_run_path
from ert.runpaths import Runpaths
from ert.storage import open_storage


@pytest.mark.parametrize(
    "distribution_settings",
    [
        pytest.param({"name": "uniform", "min": 0.0, "max": 1.0}, id="uniform"),
        pytest.param({"name": "logunif", "min": 1e-5, "max": 1.0}, id="logunif"),
        pytest.param(
            {"name": "dunif", "steps": 1000, "min": 0.0, "max": 1.0},
            id="dunif",
        ),
        pytest.param({"name": "normal", "mean": 0.0, "std": 1.0}, id="normal"),
        pytest.param({"name": "lognormal", "mean": 0.0, "std": 1.0}, id="lognormal"),
        pytest.param(
            {
                "name": "truncated_normal",
                "mean": 0.0,
                "std": 1.0,
                "min": -2.0,
                "max": 2.0,
            },
            id="truncated_normal",
        ),
        pytest.param({"name": "raw"}, id="raw"),
        pytest.param({"name": "const", "value": 0.0}, id="const"),
        pytest.param(
            {"name": "triangular", "min": 0.0, "mode": 0.5, "max": 1.0},
            id="triangular",
        ),
        pytest.param(
            {"name": "errf", "min": 0.0, "max": 1.0, "skewness": 0.0, "width": 1.0},
            id="errf",
        ),
        pytest.param(
            {
                "name": "derrf",
                "steps": 1000,
                "min": 0.0,
                "max": 1.0,
                "skewness": 0.0,
                "width": 1.0,
            },
            id="derrf",
        ),
    ],
)
def test_create_run_path_load_scalar_keys_performance(
    benchmark, tmp_path, distribution_settings
):
    reals = 100
    num_keys = 100
    storage_path = tmp_path / "storage"

    parameter_configs = [
        GenKwConfig(
            name=f"P{i}",
            distribution=distribution_settings,
        )
        for i in range(num_keys)
    ]

    with open_storage(storage_path, mode="w") as storage:
        exp = storage.create_experiment(
            experiment_config={
                "parameter_configs": [
                    pc.model_dump(mode="json") for pc in parameter_configs
                ]
            },
            name="perf-exp",
        )
        ensemble = exp.create_ensemble(ensemble_size=reals, name="default")

        rng = np.random.default_rng(42)
        values = rng.standard_normal(size=(reals, num_keys)).astype(np.float32)
        df = pl.DataFrame(values, schema=[cfg.name for cfg in parameter_configs])
        df = df.insert_column(
            0, pl.Series("realization", np.arange(reals, dtype=np.int32))
        )
        df.write_parquet(ensemble.mount_point / "SCALAR.parquet")

        active = [True] * reals
        counter = itertools.count()

        def run():
            idx = next(counter)
            runpath_root = tmp_path / f"runpaths_{idx}"
            runpaths = Runpaths(
                jobname_format="job_<IENS>",
                runpath_format=str(
                    runpath_root / "realization-<IENS>" / "iteration-<ITER>"
                ),
                filename=runpath_root / ".ert_runpath_list",
            )
            run_args = create_run_arguments(runpaths, active, ensemble)

            create_run_path(
                run_args=run_args,
                ensemble=ensemble,
                user_config_file="perf.ert",
                env_vars={},
                env_pr_fm_step={},
                forward_model_steps=[],
                substitutions={},
                parameters_file="parameters",
                runpaths=runpaths,
            )

        benchmark(run)
