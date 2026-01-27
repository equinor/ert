from pathlib import Path
from textwrap import dedent

import pytest

from ert.config import ErtConfig
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.storage import open_storage
from tests.ert.ui_tests.cli.run_cli import run_cli


@pytest.mark.timeout(600)
@pytest.mark.usefixtures("copy_snake_oil_field")
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
        VALUE   = 0.5;
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

    prior_mean = poro_prior["values"].mean(dim="realizations")
    posterior_mean = poro_posterior["values"].mean(dim="realizations")
    delta_mean = posterior_mean - prior_mean
    obs_col_update = delta_mean.sel(x=5, y=5)  # location of observation
    obs_update_obs = obs_col_update.mean(dim="z")
    obs_update_global = delta_mean.mean(dim="z").mean(dim="y").mean(dim="x")
    assert obs_update_obs > obs_update_global
