import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from ecl.summary import EclSum

from ert import LibresFacade
from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert.analysis import ErtAnalysisError, ESUpdate


@pytest.fixture
def prior_ensemble(storage, setup_configuration):
    ert_config = setup_configuration.ert_config
    return storage.create_experiment(
        parameters=ert_config.ensemble_config.parameter_configuration
    ).create_ensemble(ensemble_size=100, name="prior")


@pytest.fixture
def target_ensemble(storage):
    return storage.create_experiment().create_ensemble(ensemble_size=100, name="target")


@pytest.fixture
def setup_configuration(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 3
        ECLBASE ECLIPSE_CASE_%d
        REFCASE ECLIPSE_CASE
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
        # We create a reference case
        ref_time = [datetime(2014, 9, 9) + timedelta(days=i) for i in range(10)]
        run_sim(ref_time, 0.5)
        ert_config = ErtConfig.from_file("config.ert")
        ert = EnKFMain(ert_config)
        yield ert


def create_responses(ert, prior_ensemble, response_times):
    cwd = Path().absolute()
    rng = np.random.default_rng(seed=1234)
    for i, response_time in enumerate(response_times):
        sim_path = cwd / "simulations" / f"realization-{i}" / "iter-0"
        sim_path.mkdir(parents=True, exist_ok=True)
        os.chdir(sim_path)
        run_sim(response_time, rng.standard_normal(), fname=f"ECLIPSE_CASE_{i}")
    os.chdir(cwd)
    facade = LibresFacade(ert)
    facade.load_from_forward_model(
        prior_ensemble, [True] * facade.get_ensemble_size(), 0
    )


def test_that_reading_matching_time_is_ok(
    setup_configuration, prior_ensemble, target_ensemble
):
    ert = setup_configuration
    ert.sample_prior(prior_ensemble, list(range(ert.getEnsembleSize())))

    create_responses(
        ert, prior_ensemble, ert.getEnsembleSize() * [[datetime(2014, 9, 9)]]
    )

    es_update = ESUpdate(ert)

    es_update.smootherUpdate(prior_ensemble, target_ensemble, "an id")


def test_that_mismatched_responses_give_error(
    setup_configuration, prior_ensemble, target_ensemble
):
    ert = setup_configuration
    ert.sample_prior(prior_ensemble, list(range(ert.getEnsembleSize())))

    response_times = [
        [datetime(2014, 9, 9)],
        [datetime(2014, 9, 9)],
        [datetime(2017, 9, 9)],
    ]
    create_responses(ert, prior_ensemble, response_times)

    es_update = ESUpdate(ert)

    with pytest.raises(ErtAnalysisError, match=re.escape("No active observations")):
        es_update.smootherUpdate(prior_ensemble, target_ensemble, "an id")


def test_that_different_length_is_ok_as_long_as_observation_time_exists(
    setup_configuration,
    prior_ensemble,
    target_ensemble,
):
    ert = setup_configuration
    ert.sample_prior(prior_ensemble, list(range(ert.getEnsembleSize())))
    response_times = [
        [datetime(2014, 9, 9)],
        [datetime(2014, 9, 9)],
        [datetime(2014, 9, 9), datetime(2017, 9, 9)],
        [datetime(2014, 9, 9)],
        [datetime(2014, 9, 9), datetime(1988, 9, 9)],
    ]
    create_responses(ert, prior_ensemble, response_times)

    es_update = ESUpdate(ert)

    es_update.smootherUpdate(prior_ensemble, target_ensemble, "an id")


def run_sim(dates, value, fname="ECLIPSE_CASE"):
    """
    Create a summary file, the contents of which are not important
    """
    start_date = dates[0]
    ecl_sum = EclSum.writer(fname, start_date, 3, 3, 3)
    ecl_sum.addVariable("FOPR", unit="SM3/DAY")
    for report_step, date in enumerate(dates):
        t_step = ecl_sum.addTStep(
            report_step + 1, sim_days=(date + timedelta(days=1) - start_date).days
        )
        t_step["FOPR"] = value
    ecl_sum.fwrite()
