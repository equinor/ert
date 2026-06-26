import json
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from queue import SimpleQueue
from textwrap import dedent
from unittest.mock import MagicMock

import pytest
from polars import Float32, Series
from polars.testing import assert_series_equal

from ert.config import ErtConfig
from ert.mode_definitions import (
    ENSEMBLE_EXPERIMENT_MODE,
    ES_MDA_MODE,
)
from ert.run_models import ErtRunError, create_model
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
    def observations(*, restart: bool = False):
        if restart:
            return dedent(
                """
                GENERAL_OBSERVATION POLY_OBS {
                    DATA       = POLY_RES;
                    INDEX_LIST = 0,2,4,6,8,10;
                    OBS_FILE   = obs_data.txt;
                };
                """
            )
        return dedent(
            """
            GENERAL_OBSERVATION POLY_OBS {
                DATA       = POLY_RES;
                INDEX_LIST = 0,2,4,6,8;
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
        9.0 1.5
        15.0 3.0
        30.0 6.0
        50.0 12.0
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
        "--weights=2,1",
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
        Series("observations", [2.0, 9.0, 15.0, 30.0, 50.0], dtype=Float32),
    )

    assert_series_equal(
        experiment.observations["gen_data"]["std"],
        Series("std", [0.5, 1.5, 3.0, 6.0, 12.0], dtype=Float32),
    )


@pytest.fixture
def poly_prior_ensemble_id(
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
        ensemble = storage.get_experiment_by_name(
            "ensemble-experiment"
        ).get_ensemble_by_name("default")
        return str(ensemble.id)


def _build_esmda_restart_model(prior_ensemble_id: str):
    config = ErtConfig.from_file("config.ert")
    return create_model(
        config,
        Namespace(
            mode=ES_MDA_MODE,
            realizations=None,
            target_ensemble="iter-<ITER>",
            weights="1,1",
            restart_run=True,
            prior_ensemble_id=prior_ensemble_id,
            experiment_name="restart-experiment",
        ),
        SimpleQueue(),
    )


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_restarting_esmda_with_invalid_prior_ensemble_id_gives_error_message(
    poly_prior_ensemble_id,
):
    model = _build_esmda_restart_model(poly_prior_ensemble_id)

    model.prior_ensemble_id = "not-a-valid-uuid"

    with pytest.raises(
        ErtRunError,
        match="Prior ensemble with ID: not-a-valid-uuid does not exist or is broken",
    ):
        model.run_experiment(MagicMock())


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_restarting_esmda_from_prior_with_localized_gen_obs_gives_error_message(
    poly_prior_ensemble_id,
):
    with open_storage("storage", mode="w") as storage:
        experiment = storage.get_experiment_by_name("ensemble-experiment")
        index_path = experiment._path / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))

        # Add invalid keys to observation to trigger validation failure
        for observation in index["experiment"]["observations"]:
            observation["east"] = None
            observation["north"] = None
            observation["radius"] = None
        index_path.write_text(json.dumps(index), encoding="utf-8")

    model = _build_esmda_restart_model(poly_prior_ensemble_id)

    try:
        with pytest.raises(
            ErtRunError,
            match=(
                f"Could not restart from prior ensemble 'default' "
                f"\\(ID: {poly_prior_ensemble_id}\\):"
            ),
        ):
            model.run_experiment(MagicMock())
    finally:
        model._clean_env_context()
