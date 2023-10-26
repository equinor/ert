import fileinput
import os
from pathlib import Path

import pytest

from ert.config import ErtConfig
from ert.enkf_main import create_run_path, ensemble_context, sample_prior



def test_with_gen_kw(snake_oil_case, storage):
    main = snake_oil_case

    experiment_id = storage.create_experiment(
        parameters=main.ensembleConfig().parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=1
    )
    prior = ensemble_context(
        prior_ensemble,
        [True],
        0,
        None,
        "",
        ert_config.model_config.runpath_format_string,
        "name",
    )
    sample_prior(prior_ensemble, [0])
    create_run_path(prior, ert_config.substitution_list, ert_config)
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
    prior = ensemble_context(
        prior_ensemble,
        [True],
        0,
        None,
        "",
        ert_config.model_config.runpath_format_string,
        "name",
    )
    sample_prior(prior_ensemble, [0])
    create_run_path(prior, ert_config.substitution_list, ert_config)
    assert os.path.exists("storage/snake_oil/runpath/realization-0/iter-0")
    assert not os.path.exists(
        "storage/snake_oil/runpath/realization-0/iter-0/parameters.txt"
    )
    assert len(os.listdir("storage/snake_oil/runpath")) == 1
    assert len(os.listdir("storage/snake_oil/runpath/realization-0")) == 1


def test_jobs_file_is_backed_up(snake_oil_case, storage):

    experiment_id = storage.create_experiment(
        parameters=main.ensembleConfig().parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=5
    )
    prior = ensemble_context(
        prior_ensemble,
        [True],
        0,
        None,
        "",
        ert_config.model_config.runpath_format_string,
        "name",
    )
    sample_prior(prior_ensemble, [0])
    create_run_path(prior, ert_config.substitution_list, ert_config)
    assert os.path.exists("storage/snake_oil/runpath/realization-0/iter-0/jobs.json")
    create_run_path(prior, ert_config.substitution_list, ert_config)
    iter0_output_files = os.listdir("storage/snake_oil/runpath/realization-0/iter-0/")
    jobs_files = [f for f in iter0_output_files if f.startswith("jobs.json")]
    assert len(jobs_files) > 1, "No backup created for jobs.json"
