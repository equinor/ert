import os
from pathlib import Path

import pytest

from ert.config import ErtConfig
from ert.enkf_main import create_run_path, sample_prior

config_contents = """\
NUM_REALIZATIONS 1
QUEUE_SYSTEM LOCAL
ENSPATH storage
{parameters}
"""


@pytest.fixture
def make_run_path(run_paths, run_args, storage):
    def func(ert_config):
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
            env_pr_fm_step=ert_config.env_pr_fm_step,
            substitutions=ert_config.substitutions,
            templates=ert_config.ert_templates,
            model_config=ert_config.model_config,
            runpaths=run_paths(ert_config),
        )

    return func


@pytest.mark.usefixtures("use_tmpdir")
def test_setup_with_gen_kw_generates_parameters_txt(make_run_path):
    Path("genkw").write_text("genkw0 UNIFORM 0 1", encoding="utf-8")
    ert_config = ErtConfig.from_file_contents(
        config_contents.format(parameters="GEN_KW GENKW genkw")
    )
    make_run_path(ert_config)
    assert os.path.exists("simulations/realization-0/iter-0")
    assert os.path.exists("simulations/realization-0/iter-0/parameters.txt")
    assert len(os.listdir("simulations")) == 1
    assert len(os.listdir("simulations/realization-0")) == 1


@pytest.mark.usefixtures("use_tmpdir")
def test_setup_without_gen_kw_does_not_generates_parameters_txt(make_run_path):
    ert_config = ErtConfig.from_file_contents(config_contents.format(parameters=""))
    make_run_path(ert_config)
    assert os.path.exists("simulations/realization-0/iter-0")
    assert not os.path.exists("simulations/realization-0/iter-0/parameters.txt")
    assert len(os.listdir("simulations")) == 1
    assert len(os.listdir("simulations/realization-0")) == 1


@pytest.mark.usefixtures("use_tmpdir")
def test_jobs_file_is_backed_up(make_run_path):
    Path("genkw").write_text("genkw0 UNIFORM 0 1", encoding="utf-8")
    ert_config = ErtConfig.from_file_contents(
        config_contents.format(parameters="GEN_KW GENKW genkw")
    )
    make_run_path(ert_config)
    assert os.path.exists("simulations/realization-0/iter-0/jobs.json")
    make_run_path(ert_config)
    iter0_output_files = os.listdir("simulations/realization-0/iter-0/")
    assert (
        len([f for f in iter0_output_files if f.startswith("jobs.json")]) > 1
    ), "No backup created for jobs.json"
