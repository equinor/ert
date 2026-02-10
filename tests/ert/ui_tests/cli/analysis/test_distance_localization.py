import pytest

from ert.config import ErtConfig
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.storage import open_storage
from tests.ert.ui_tests.cli.run_cli import run_cli


def assert_variance_in_field(
    prior_ensemble, posterior_ensemble, field_name, obs_pos, no_upd_pos
):
    prior_data = prior_ensemble.load_parameters(field_name)
    posterior_data = posterior_ensemble.load_parameters(field_name)

    prior_var = prior_data.var(dim="realizations", ddof=1).sel(
        x=obs_pos[0], y=obs_pos[1]
    )
    posterior_var = posterior_data.var(dim="realizations", ddof=1).sel(
        x=obs_pos[0], y=obs_pos[1]
    )

    assert prior_var.mean() >= posterior_var.mean(), (
        f"Expected variance reduction at position {obs_pos} in field {field_name}, "
    )


@pytest.mark.timeout(600)
@pytest.mark.usefixtures("copy_snake_oil_field")
@pytest.mark.integration_test
def test_that_distance_localization_works_with_a_single_observation():
    with open("snake_oil_field.ert", "r+", encoding="utf-8") as f:
        lines = f.readlines()

    config_content = [
        line for line in lines if not line.lstrip().startswith("OBS_CONFIG")
    ]
    config_content.extend(
        [
            "ANALYSIS_SET_VAR STD_ENKF DISTANCE_LOCALIZATION True\n",
            "OBS_CONFIG observations/observations_loc.txt\n",
        ]
    )

    with open("snake_oil_field_dl.ert", "w", encoding="utf-8") as f:
        f.writelines(config_content)

    run_cli(
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitoring",
        "snake_oil_field_dl.ert",
        "--experiment-name",
        "dl",
    )

    ert_config = ErtConfig.from_file("snake_oil_field_dl.ert")
    assert ert_config.analysis_config.es_settings.distance_localization is True
    storage = open_storage(ert_config.ens_path)
    experiment = storage.get_experiment_by_name("dl")
    ens_prior = experiment.get_ensemble_by_name("iter-0")
    ens_posterior = experiment.get_ensemble_by_name("iter-1")

    assert_variance_in_field(ens_prior, ens_posterior, "PORO", (5, 5), (9, 9))


@pytest.mark.timeout(600)
@pytest.mark.usefixtures("copy_heat_equation")
@pytest.mark.integration_test
def test_that_distance_localization_runs_on_heat_equation():
    with open("config.ert", encoding="utf-8") as fh:
        lines = fh.readlines()

    config_content = [
        line for line in lines if not line.lstrip().startswith("ANALYSIS_SET_VAR")
    ]
    config_content.extend(
        [
            "ANALYSIS_SET_VAR STD_ENKF DISTANCE_LOCALIZATION True\n",
            "ENSPATH heat_storage_dl\n",
        ]
    )
    with open("heat_dl.ert", "w", encoding="utf-8") as fh:
        fh.writelines(config_content)

    run_cli(
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitoring",
        "heat_dl.ert",
        "--experiment-name",
        "heat_dl",
    )

    config = ErtConfig.from_file("heat_dl.ert")
    assert config.analysis_config.es_settings.distance_localization is True
    with open_storage(config.ens_path) as storage:
        experiment = storage.get_experiment_by_name("heat_dl")
        prior = experiment.get_ensemble_by_name("iter-0")
        posterior = experiment.get_ensemble_by_name("iter-1")
        assert_variance_in_field(prior, posterior, "COND", (2, 2), (9, 9))
