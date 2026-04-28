from pathlib import Path

import numpy as np
import pytest

from ert.config import ErtConfig
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.storage import open_storage
from tests.ert.ui_tests.cli.run_cli import run_cli


def assert_stronger_variance_reduction_at_observation_location(
    prior_ensemble, posterior_ensemble, field_name, obs_pos, unobserved_pos
):
    prior_data = prior_ensemble.load_parameters(field_name)
    posterior_data = posterior_ensemble.load_parameters(field_name)

    prior_var_obs = prior_data.var(dim="realizations", ddof=1).sel(
        x=obs_pos[0], y=obs_pos[1]
    )
    posterior_var_obs = posterior_data.var(dim="realizations", ddof=1).sel(
        x=obs_pos[0], y=obs_pos[1]
    )

    prior_var_no_obs = prior_data.var(dim="realizations", ddof=1).sel(
        x=unobserved_pos[0], y=unobserved_pos[1]
    )
    posterior_var_no_obs = posterior_data.var(dim="realizations", ddof=1).sel(
        x=unobserved_pos[0], y=unobserved_pos[1]
    )

    obs_reduction = prior_var_obs.mean() - posterior_var_obs.mean()
    no_obs_reduction = prior_var_no_obs.mean() - posterior_var_no_obs.mean()

    assert obs_reduction > no_obs_reduction, (
        f"Expecting stronger variance reduction at observation location "
        f"than outside on {field_name}"
    )


@pytest.mark.timeout(600)
@pytest.mark.usefixtures("copy_heat_equation")
@pytest.mark.slow
def test_that_distance_localization_reduces_posterior_variance():
    with Path("config.ert").open(encoding="utf-8") as fh:
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
    with Path("heat_dl.ert").open("w", encoding="utf-8") as fh:
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

        assert_stronger_variance_reduction_at_observation_location(
            prior, posterior, "COND", obs_pos=(25, 25), unobserved_pos=(5, 5)
        )

        df_scalars_prior = prior.load_scalars()
        df_scalars_posterior = posterior.load_scalars()

        assert (
            0
            < np.linalg.det(
                np.cov(
                    df_scalars_posterior.drop("realization").to_numpy(), rowvar=False
                )
            )
            < np.linalg.det(
                np.cov(df_scalars_prior.drop("realization").to_numpy(), rowvar=False)
            )
        )

        prior_cond = prior.load_parameters("COND")
        posterior_cond = posterior.load_parameters("COND")

        param_config = config.ensemble_config.parameter_configs["COND"]

        prior_cov = np.cov(
            prior_cond["values"]
            .to_numpy()
            .reshape(
                prior.ensemble_size,
                param_config.nx * param_config.ny * param_config.nz,
            ),
            rowvar=False,
        )
        posterior_cov = np.cov(
            posterior_cond["values"]
            .to_numpy()
            .reshape(
                posterior.ensemble_size,
                param_config.nx * param_config.ny * param_config.nz,
            ),
            rowvar=False,
        )

        assert np.trace(posterior_cov) < np.trace(prior_cov)

        obs_keys = list(experiment.observation_keys)

        def mean_normalized_misfit(ensemble) -> float:
            real_indices = np.array(ensemble.get_realization_list_with_responses())
            df = ensemble.get_observations_and_responses(obs_keys, real_indices)
            obs = df["observations"].to_numpy()
            std = df["std"].to_numpy()
            realization_cols = [str(r) for r in real_indices]
            simulated = df.select(realization_cols).to_numpy()
            normalized_residuals = (
                (simulated - obs[:, np.newaxis]) / std[:, np.newaxis]
            ) ** 2
            return float(normalized_residuals.mean())

        assert mean_normalized_misfit(posterior) < mean_normalized_misfit(prior), (
            "Expected posterior responses to have a lower normalized misfit "
            "against observations than the prior"
        )
