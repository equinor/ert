import fileinput
import os
from pathlib import Path

import pytest

from ert.config import ErtConfig
from ert.enkf_main import EnKFMain, sample_prior


def test_with_gen_kw(copy_case, storage):
    copy_case("snake_oil")
    ert_config = ErtConfig.from_file("snake_oil.ert")
    main = EnKFMain(ert_config)
    experiment_id = storage.create_experiment(
        parameters=ert_config.ensemble_config.parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=main.getEnsembleSize()
    )
    prior = main.ensemble_context(prior_ensemble, [True], 0)
    sample_prior(prior_ensemble, [0])
    main.createRunPath(prior)
    assert os.path.exists(
        "storage/snake_oil/runpath/realization-0/iter-0/parameters.txt"
    )
    assert len(os.listdir("storage/snake_oil/runpath")) == 1
    assert len(os.listdir("storage/snake_oil/runpath/realization-0")) == 1


@pytest.mark.usefixtures("copy_snake_oil_case")
def test_without_gen_kw(prior_ensemble):
    with fileinput.input("snake_oil.ert", inplace=True) as fin:
        for line in fin:
            if line.startswith("GEN_KW"):
                continue
            print(line, end="")
    assert "GEN_KW" not in Path("snake_oil.ert").read_text("utf-8")
    ert_config = ErtConfig.from_file("snake_oil.ert")
    main = EnKFMain(ert_config)
    prior = main.ensemble_context(prior_ensemble, [True], 0)
    sample_prior(prior_ensemble, [0])
    main.createRunPath(prior)
    assert os.path.exists("storage/snake_oil/runpath/realization-0/iter-0")
    assert not os.path.exists(
        "storage/snake_oil/runpath/realization-0/iter-0/parameters.txt"
    )
    assert len(os.listdir("storage/snake_oil/runpath")) == 1
    assert len(os.listdir("storage/snake_oil/runpath/realization-0")) == 1


def test_jobs_file_is_backed_up(copy_case, storage):
    copy_case("snake_oil")
    ert_config = ErtConfig.from_file("snake_oil.ert")
    main = EnKFMain(ert_config)
    experiment_id = storage.create_experiment(
        parameters=ert_config.ensemble_config.parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=5
    )
    prior = main.ensemble_context(prior_ensemble, [True], 0)
    sample_prior(prior_ensemble, [0])
    main.createRunPath(prior)
    assert os.path.exists("storage/snake_oil/runpath/realization-0/iter-0/jobs.json")
    main.createRunPath(prior)
    iter0_output_files = os.listdir("storage/snake_oil/runpath/realization-0/iter-0/")
    jobs_files = [f for f in iter0_output_files if f.startswith("jobs.json")]
    assert len(jobs_files) > 1, "No backup created for jobs.json"
