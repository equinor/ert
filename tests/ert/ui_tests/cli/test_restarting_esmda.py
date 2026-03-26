from collections.abc import Callable
from pathlib import Path
from textwrap import dedent

import pytest
from polars import Float32, Series
from polars.testing import assert_series_equal

from ert.mode_definitions import (
    ENSEMBLE_EXPERIMENT_MODE,
    ES_MDA_MODE,
)
from ert.storage import open_storage

from .run_cli import run_cli


@pytest.fixture
def ert_config() -> str:
    return dedent(
        """
        QUEUE_SYSTEM LOCAL

        RUNPATH poly_out/realization-<IENS>/iter-<ITER>

        OBS_CONFIG observations

        NUM_REALIZATIONS 2

        GEN_KW COEFFS coeff_priors
        GEN_DATA POLY_RES RESULT_FILE:ert.out

        INSTALL_JOB ert_eval ERT_EVAL
        FORWARD_MODEL ert_eval
        """
    )


@pytest.fixture
def parameters() -> str:
    return dedent(
        """
        a UNIFORM 0 1
        b UNIFORM 0 2
        c UNIFORM 0 5
        """
    )


@pytest.fixture
def poly_eval() -> str:
    return dedent(
        """
        EXECUTABLE ert_eval.py
        """
    )


@pytest.fixture
def observations() -> Callable[[bool], str]:
    def observations(restart: bool = False):
        if restart:
            return dedent(
                """
                GENERAL_OBSERVATION POLY_OBS {
                    DATA       = POLY_RES;
                    INDEX_LIST = 0,2;
                    OBS_FILE   = obs_data.txt;
                };
                """
            )
        return dedent(
            """
            GENERAL_OBSERVATION POLY_OBS {
                DATA       = POLY_RES;
                INDEX_LIST = 0;
                OBS_FILE   = obs_data.txt;
            };
            """
        )

    return observations


@pytest.fixture
def observations_data() -> str:
    return dedent(
        """
        2.0 0.5
        """
    )


@pytest.fixture
def ert_script() -> str:
    script = r"""
#!/usr/bin/env python3
import json
from pathlib import Path

def _evaluate(coeffs, t):
    return coeffs["a"]["value"] * t**2 + coeffs["b"]["value"] * t + coeffs["c"]["value"]

if __name__ == "__main__":
    coeffs = json.loads(Path("parameters.json").read_text(encoding="utf-8"))
    output = [_evaluate(coeffs, t) for t in range(10)]
    Path("ert.out").write_text("\n".join(map(str, output)), encoding="utf-8")
"""
    return script.removeprefix("\n")


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_running_esmda_from_restart_uses_previous_observations_and_parameters(
    use_tmpdir,
    ert_config,
    parameters,
    poly_eval,
    ert_script,
    observations,
    observations_data,
):
    Path("config.ert").write_text(ert_config, encoding="utf-8")
    Path("coeff_priors").write_text(parameters, encoding="utf-8")
    Path("ERT_EVAL").write_text(poly_eval, encoding="utf-8")
    Path("observations").write_text(observations(), encoding="utf-8")
    Path("obs_data.txt").write_text(observations_data, encoding="utf-8")
    Path("ert_eval.py").write_text(ert_script, encoding="utf-8")
    Path("ert_eval.py").chmod(0o755)

    run_cli(
        ENSEMBLE_EXPERIMENT_MODE,
        "--disable-monitoring",
        "config.ert",
    )

    with open_storage("storage") as storage:
        experiment = storage.get_experiment_by_name("ensemble-experiment")
        ensemble = experiment.get_ensemble_by_name("default")

    for param in experiment.parameter_keys:
        assert param in {"a", "b", "c"}

    # Update parameters and observations and restart with es-mda.
    Path("coeff_priors").write_text(parameters + "d UNIFORM 0 5\n", encoding="utf-8")
    Path("obs_data.txt").write_text(
        observations_data + "100.0 24.0\n", encoding="utf-8"
    )
    Path("observations").write_text(observations(restart=True), encoding="utf-8")

    run_cli(
        ES_MDA_MODE,
        "--disable-monitoring",
        "--weights=0,1",
        "--restart-ensemble-id",
        str(ensemble.id),
        "config.ert",
    )

    with open_storage("storage") as storage:
        experiment = storage.get_experiment_by_name("Restart from default")

    for param in experiment.parameter_keys:
        assert param in {"a", "b", "c"}

    assert "d" not in experiment.parameter_keys

    assert_series_equal(
        experiment.observations["gen_data"]["observations"],
        Series("observations", [2.0], dtype=Float32),
    )

    assert_series_equal(
        experiment.observations["gen_data"]["std"],
        Series("std", [0.5], dtype=Float32),
    )
