import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from resdata.summary import Summary

from ert import LibresFacade
from ert.analysis import ErtAnalysisError, UpdateConfiguration, smoother_update
from ert.config import ErtConfig
from ert.enkf_main import sample_prior


@pytest.fixture
def prior_ensemble(storage, ert_config):
    return storage.create_experiment(
        parameters=ert_config.ensemble_config.parameter_configuration,
        responses=ert_config.ensemble_config.response_configuration,
        observations=ert_config.observations,
    ).create_ensemble(ensemble_size=3, name="prior")


@pytest.fixture
def ert_config(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 3
        ECLBASE ECLIPSE_CASE_%d
        OBS_CONFIG observations
        GEN_KW KW_NAME template.txt kw.txt prior.txt
        RANDOM_SEED 1234
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fh:
            obs_config = dedent(
                """
                SUMMARY_OBSERVATION FOPR_1
                {
                VALUE   = 0.9;
                ERROR   = 0.05;
                DATE    = 2014-09-10;
                KEY     = FOPR;
                };
                SUMMARY_OBSERVATION FOPR_2
                {
                VALUE   = 1.1;
                ERROR   = 0.05;
                DATE    = 2014-09-11;
                KEY     = FOPR;
                };
                """
            )
            fh.writelines(obs_config)
        with open("template.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")
        yield ErtConfig.from_file("config.ert")


def create_responses(config_file, prior_ensemble, response_times):
    cwd = Path().absolute()
    rng = np.random.default_rng(seed=1234)
    for i, response_time in enumerate(response_times):
        sim_path = cwd / "simulations" / f"realization-{i}" / "iter-0"
        sim_path.mkdir(parents=True, exist_ok=True)
        os.chdir(sim_path)
        run_sim(response_time, rng.standard_normal(), fname=f"ECLIPSE_CASE_{i}")
    os.chdir(cwd)
    facade = LibresFacade.from_config_file(config_file)
    facade.load_from_forward_model(
        prior_ensemble, [True] * facade.get_ensemble_size(), 0
    )


def test_that_reading_matching_time_is_ok(ert_config, storage, prior_ensemble):
    sample_prior(prior_ensemble, range(prior_ensemble.ensemble_size))

    create_responses(
        ert_config.user_config_file,
        prior_ensemble,
        ert_config.model_config.num_realizations * [[datetime(2014, 9, 9)]],
    )

    target_ensemble = storage.create_ensemble(
        prior_ensemble.experiment_id,
        ensemble_size=ert_config.model_config.num_realizations,
        iteration=1,
        name="new_ensemble",
        prior_ensemble=prior_ensemble,
    )

    smoother_update(
        prior_ensemble,
        target_ensemble,
        "an id",
        UpdateConfiguration.global_update_step(
            list(ert_config.observations.keys()),
            ert_config.ensemble_config.parameters,
        ),
    )


def test_that_mismatched_responses_give_error(ert_config, storage, prior_ensemble):
    sample_prior(prior_ensemble, range(prior_ensemble.ensemble_size))

    response_times = [
        [datetime(2014, 9, 9)],
        [datetime(2014, 9, 9)],
        [datetime(2017, 9, 9)],
    ]
    create_responses(ert_config.user_config_file, prior_ensemble, response_times)

    target_ensemble = storage.create_ensemble(
        prior_ensemble.experiment_id,
        ensemble_size=ert_config.model_config.num_realizations,
        iteration=1,
        name="new_ensemble",
        prior_ensemble=prior_ensemble,
    )

    with pytest.raises(ErtAnalysisError, match=re.escape("No active observations")):
        smoother_update(
            prior_ensemble,
            target_ensemble,
            "an id",
            UpdateConfiguration.global_update_step(
                list(ert_config.observations.keys()),
                ert_config.ensemble_config.parameters,
            ),
        )


def test_that_different_length_is_ok_as_long_as_observation_time_exists(
    ert_config,
    storage,
    prior_ensemble,
):
    sample_prior(prior_ensemble, range(prior_ensemble.ensemble_size))
    response_times = [
        [datetime(2014, 9, 9)],
        [datetime(2014, 9, 9)],
        [datetime(2014, 9, 9), datetime(2017, 9, 9)],
        [datetime(2014, 9, 9)],
        [datetime(2014, 9, 9), datetime(1988, 9, 9)],
    ]
    create_responses(ert_config.user_config_file, prior_ensemble, response_times)

    target_ensemble = storage.create_ensemble(
        prior_ensemble.experiment_id,
        ensemble_size=ert_config.model_config.num_realizations,
        iteration=1,
        name="new_ensemble",
        prior_ensemble=prior_ensemble,
    )

    smoother_update(
        prior_ensemble,
        target_ensemble,
        "an id",
        UpdateConfiguration.global_update_step(
            list(ert_config.observations.keys()),
            ert_config.ensemble_config.parameters,
        ),
    )


def run_sim(dates, value, fname="ECLIPSE_CASE"):
    """
    Create a summary file, the contents of which are not important
    """
    start_date = dates[0]
    summary = Summary.writer(fname, start_date, 3, 3, 3)
    summary.add_variable("FOPR", unit="SM3/DAY")
    for report_step, date in enumerate(dates):
        t_step = summary.add_t_step(
            report_step + 1, sim_days=(date + timedelta(days=1) - start_date).days
        )
        t_step["FOPR"] = value
    summary.fwrite()


def test_that_duplicate_summary_time_steps_does_not_fail(
    ert_config,
    storage,
    prior_ensemble,
):
    sample_prior(prior_ensemble, range(prior_ensemble.ensemble_size))
    response_times = [
        [datetime(2014, 9, 9)],
        [datetime(2014, 9, 9)],
        [datetime(2014, 9, 9), datetime(2014, 9, 9)],
        [datetime(2014, 9, 9)],
        [datetime(2014, 9, 9), datetime(1988, 9, 9)],
    ]
    create_responses(ert_config.user_config_file, prior_ensemble, response_times)

    target_ensemble = storage.create_ensemble(
        prior_ensemble.experiment_id,
        ensemble_size=ert_config.model_config.num_realizations,
        iteration=1,
        name="new_ensemble",
        prior_ensemble=prior_ensemble,
    )

    smoother_update(
        prior_ensemble,
        target_ensemble,
        "an id",
        UpdateConfiguration.global_update_step(
            list(ert_config.observations.keys()),
            ert_config.ensemble_config.parameters,
        ),
    )
