import fileinput
import os
from pathlib import Path

import pytest

from ert.config import ErtConfig
from ert.enkf_main import create_run_path, sample_prior


@pytest.mark.usefixtures("copy_snake_oil_case")
def test_with_gen_kw(storage, run_paths, run_args):
    ert_config = ErtConfig.from_file("snake_oil.ert")
    experiment_id = storage.create_experiment(
        parameters=ert_config.ensemble_config.parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=1
    )

    sample_prior(prior_ensemble, [0])
    create_run_path(
        run_args=run_args(ert_config, prior_ensemble, 1),
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths(ert_config),
    )
    assert os.path.exists(
        "storage/snake_oil/runpath/realization-0/iter-0/parameters.txt"
    )
    assert len(os.listdir("storage/snake_oil/runpath")) == 1
    assert len(os.listdir("storage/snake_oil/runpath/realization-0")) == 1


@pytest.mark.usefixtures("copy_snake_oil_case")
def test_without_gen_kw(prior_ensemble, run_args, run_paths):
    with fileinput.input("snake_oil.ert", inplace=True) as fin:
        for line in fin:
            if line.startswith("GEN_KW"):
                continue
            print(line, end="")
    assert "GEN_KW" not in Path("snake_oil.ert").read_text("utf-8")
    ert_config = ErtConfig.from_file("snake_oil.ert")
    sample_prior(prior_ensemble, [0])
    create_run_path(
        run_args=run_args(ert_config, prior_ensemble, 1),
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths(ert_config),
    )
    assert os.path.exists("storage/snake_oil/runpath/realization-0/iter-0")
    assert not os.path.exists(
        "storage/snake_oil/runpath/realization-0/iter-0/parameters.txt"
    )
    assert len(os.listdir("storage/snake_oil/runpath")) == 1
    assert len(os.listdir("storage/snake_oil/runpath/realization-0")) == 1


@pytest.mark.usefixtures("copy_snake_oil_case")
def test_jobs_file_is_backed_up(storage, run_args, run_paths):
    ert_config = ErtConfig.from_file("snake_oil.ert")
    experiment_id = storage.create_experiment(
        parameters=ert_config.ensemble_config.parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=5
    )
    run_arg = run_args(ert_config, prior_ensemble, 1)
    sample_prior(prior_ensemble, [0])
    create_run_path(
        run_args=run_arg,
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths(ert_config),
    )
    assert os.path.exists("storage/snake_oil/runpath/realization-0/iter-0/jobs.json")
    create_run_path(
        run_args=run_arg,
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths(ert_config),
    )
    iter0_output_files = os.listdir("storage/snake_oil/runpath/realization-0/iter-0/")
    jobs_files = [f for f in iter0_output_files if f.startswith("jobs.json")]
    assert len(jobs_files) > 1, "No backup created for jobs.json"
