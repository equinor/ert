from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from ert.config import ErtConfig
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE, ES_MDA_MODE
from ert.storage import open_storage
from tests.ert.ui_tests.cli.run_cli import run_cli


@pytest.mark.timeout(600)
@pytest.mark.usefixtures("copy_snake_oil_field")
@pytest.mark.integration_test
def test_that_distance_localization_works_with_a_single_observation():
    set_distance_localization = dedent(
        """
        ANALYSIS_SET_VAR STD_ENKF DISTANCE_LOCALIZATION True
        """
    )

    with open("snake_oil_field.ert", "r+", encoding="utf-8") as f:
        lines = f.readlines()
        lines.insert(9, set_distance_localization)

    with open("snake_oil_field_dl.ert", "w", encoding="utf-8") as f:
        f.writelines(lines)

    content = """
    SUMMARY_OBSERVATION WOPR_OP1_36
    {
        VALUE   = 0.7;
        ERROR   = 0.07;
        DATE    = 2010-12-26;  -- (RESTART = 36)
        KEY     = WOPR:OP1;
        LOCALIZATION {
            EAST = 5;
            NORTH = 15;
            RADIUS = 4;
        };
    };
    SUMMARY_OBSERVATION WOPR_OP1_72
    {
        VALUE   = 2.0;
        ERROR   = 0.05;
        DATE    = 2011-12-21;  -- (RESTART = 72)
        KEY     = WOPR:OP1;
        LOCALIZATION {
            EAST = 5;
            NORTH = 15;
            RADIUS = 4;
        };
    };
    """

    Path("observations/observations.txt").write_text(content, encoding="utf-8")

    run_cli(ENSEMBLE_SMOOTHER_MODE, "--disable-monitoring", "snake_oil_field_dl.ert")

    ert_config = ErtConfig.from_file("snake_oil_field_dl.ert")
    assert ert_config.analysis_config.es_settings.distance_localization is True
    storage = open_storage(ert_config.ens_path)
    experiment = storage.get_experiment_by_name("es")
    ens_prior = experiment.get_ensemble_by_name("iter-0")
    ens_posterior = experiment.get_ensemble_by_name("iter-1")

    poro_prior = ens_prior.load_parameters("PORO")
    poro_posterior = ens_posterior.load_parameters("PORO")
    prior_var = poro_prior.var(dim="realizations", ddof=1)["values"]
    post_var = poro_posterior.var(dim="realizations", ddof=1)["values"]

    sel_post = post_var.sel(x=5, y=5)
    sel_prior = prior_var.sel(x=5, y=5)

    diff = sel_prior - sel_post
    assert bool((diff > 0).all()), (
        f"Expected variance reduction, got min(prior-post)={diff.min().item()}"
    )


@pytest.mark.timeout(600)
@pytest.mark.usefixtures("copy_heat_equation")
@pytest.mark.integration_test
def test_that_distance_localization_runs_on_heat_equation():
    with open("config.ert", encoding="utf-8") as fh:
        lines = fh.readlines()

    filtered_lines = [
        line for line in lines if not line.lstrip().startswith("ANALYSIS_SET_VAR")
    ]
    filtered_lines.append(
        dedent(
            """
            ANALYSIS_SET_VAR STD_ENKF DISTANCE_LOCALIZATION True\n
            """
        )
    )
    with open("config.ert", "w", encoding="utf-8") as fh:
        fh.writelines(filtered_lines)

    run_cli(
        ES_MDA_MODE,
        "--disable-monitoring",
        "config.ert",
        "--experiment-name",
        "heat-dl",
    )

    config = ErtConfig.from_file("config.ert")
    with open_storage(config.ens_path) as storage:
        experiment = storage.get_experiment_by_name("heat-dl")
        prior = experiment.get_ensemble_by_name("default_0")
        posterior = experiment.get_ensemble_by_name("default_3")

        param_config = config.ensemble_config.parameter_configs["COND"]

        prior_result = prior.load_parameters("COND")["values"]
        prior_covariance = np.cov(
            prior_result.values.reshape(
                prior.ensemble_size, param_config.nx * param_config.ny * param_config.nz
            ),
            rowvar=False,
        )

        posterior_result = posterior.load_parameters("COND")["values"]
        posterior_covariance = np.cov(
            posterior_result.values.reshape(
                posterior.ensemble_size,
                param_config.nx * param_config.ny * param_config.nz,
            ),
            rowvar=False,
        )

        # Check that generalized variance is reduced by update step.
        assert np.trace(prior_covariance) > np.trace(posterior_covariance)
