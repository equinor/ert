import os
from pathlib import Path

import pytest

from ert.config import ErtConfig
from ert.enkf_main import create_run_path, sample_prior
from ert.run_arg import create_run_arguments


@pytest.mark.parametrize(
    "config_dict",
    [
        {"ECLBASE": "name<IENS>"},
        {"JOBNAME": "name<IENS>"},
    ],
)
def test_create_run_args(prior_ensemble, config_dict, run_paths):
    ensemble_size = 10
    config = ErtConfig.from_dict(config_dict)

    run_args = create_run_arguments(
        run_paths(config), [True] * ensemble_size, prior_ensemble
    )
    assert [real.runpath for real in run_args] == [
        f"{Path().absolute()}/simulations/realization-{i}/iter-0"
        for i in range(ensemble_size)
    ]
    assert [real.job_name for real in run_args] == [
        f"name{i}" for i in range(ensemble_size)
    ]

    substitutions = config.substitution_list
    assert "<RUNPATH>" in substitutions
    assert substitutions.get("<ECL_BASE>") == "name<IENS>"
    assert substitutions.get("<ECLBASE>") == "name<IENS>"


def test_create_run_args_separate_base_and_name(prior_ensemble, run_paths):
    ensemble_size = 10
    config = ErtConfig.from_dict({"JOBNAME": "name<IENS>", "ECLBASE": "base<IENS>"})
    run_args = create_run_arguments(
        run_paths(config), [True] * ensemble_size, prior_ensemble
    )

    assert [real.runpath for real in run_args] == [
        f"{Path().absolute()}/simulations/realization-{i}/iter-0"
        for i in range(ensemble_size)
    ]
    assert [real.job_name for real in run_args] == [
        f"name{i}" for i in range(ensemble_size)
    ]

    substitutions = config.substitution_list
    assert "<RUNPATH>" in substitutions
    assert substitutions.get("<ECL_BASE>") == "base<IENS>"
    assert substitutions.get("<ECLBASE>") == "base<IENS>"


def test_assert_symlink_deleted(snake_oil_field_example, storage, run_paths):
    ert_config = snake_oil_field_example
    experiment_id = storage.create_experiment(
        parameters=ert_config.ensemble_config.parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id,
        name="prior",
        ensemble_size=ert_config.model_config.num_realizations,
    )

    # create directory structure
    runpaths = run_paths(ert_config)
    run_args = create_run_arguments(
        runpaths,
        [True] * prior_ensemble.ensemble_size,
        prior_ensemble,
    )
    config = snake_oil_field_example
    sample_prior(prior_ensemble, range(prior_ensemble.ensemble_size))
    create_run_path(
        run_args,
        prior_ensemble,
        config,
        runpaths,
    )

    # replace field file with symlink
    linkpath = f"{run_args[0].runpath}/permx.grdecl"
    targetpath = f"{run_args[0].runpath}/permx.grdecl.target"
    with open(targetpath, "a", encoding="utf-8"):
        pass
    os.remove(linkpath)
    os.symlink(targetpath, linkpath)

    # recreate directory structure
    create_run_path(
        run_args,
        prior_ensemble,
        config,
        runpaths,
    )

    # ensure field symlink is replaced by file
    assert not os.path.islink(linkpath)


@pytest.mark.usefixtures("use_tmpdir")
def test_ert_context():
    # Write a minimal config file with DEFINE
    with open("config_file.ert", "w", encoding="utf-8") as fout:
        fout.write("NUM_REALIZATIONS 1\nDEFINE <MY_PATH> <CONFIG_PATH>")
    ert_config = ErtConfig.from_file("config_file.ert")
    context = ert_config.substitution_list
    my_path = context["<MY_PATH>"]
    assert my_path == os.getcwd()
