import os
from pathlib import Path

import pytest

from ert.config import ErtConfig
from ert.enkf_main import EnKFMain, create_run_path, ensemble_context, sample_prior


@pytest.mark.parametrize(
    "config_dict",
    [
        {"ECLBASE": "name<IENS>"},
        {"JOBNAME": "name<IENS>"},
    ],
)
def test_create_run_context(prior_ensemble, config_dict):
    iteration = 0
    ensemble_size = 10
    config = ErtConfig.from_dict(config_dict)

    run_context = ensemble_context(
        prior_ensemble,
        [True] * ensemble_size,
        iteration=iteration,
        substitution_list=None,
        jobname_format=config.model_config.jobname_format_string,
        runpath_format=config.model_config.runpath_format_string,
        runpath_file=config.runpath_file,
    )
    assert run_context.sim_fs.name == "prior"
    assert run_context.mask == [True] * ensemble_size
    assert [real.runpath for real in run_context] == [
        f"{Path().absolute()}/simulations/realization-{i}/iter-0"
        for i in range(ensemble_size)
    ]
    assert [real.job_name for real in run_context] == [
        f"name{i}" for i in range(ensemble_size)
    ]

    substitutions = config.substitution_list
    assert "<RUNPATH>" in substitutions
    assert substitutions.get("<ECL_BASE>") == "name<IENS>"
    assert substitutions.get("<ECLBASE>") == "name<IENS>"


def test_create_run_context_separate_base_and_name(monkeypatch, prior_ensemble):
    iteration = 0
    ensemble_size = 10
    config = ErtConfig.from_dict({"JOBNAME": "name<IENS>", "ECLBASE": "base<IENS>"})

    run_context = ensemble_context(
        prior_ensemble,
        [True] * ensemble_size,
        iteration=iteration,
        substitution_list=None,
        jobname_format=config.model_config.jobname_format_string,
        runpath_format=config.model_config.runpath_format_string,
        runpath_file=config.runpath_file,
    )
    assert run_context.sim_fs.name == "prior"
    assert run_context.mask == [True] * ensemble_size
    assert [real.runpath for real in run_context] == [
        f"{Path().absolute()}/simulations/realization-{i}/iter-0"
        for i in range(ensemble_size)
    ]
    assert [real.job_name for real in run_context] == [
        f"name{i}" for i in range(ensemble_size)
    ]

    substitutions = config.substitution_list
    assert "<RUNPATH>" in substitutions
    assert substitutions.get("<ECL_BASE>") == "base<IENS>"
    assert substitutions.get("<ECLBASE>") == "base<IENS>"


def test_assert_symlink_deleted(snake_oil_field_example, storage):
    ert = snake_oil_field_example
    experiment_id = storage.create_experiment(
        parameters=ert.ert_config.ensemble_config.parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=ert.getEnsembleSize()
    )

    # create directory structure
    run_context = ensemble_context(
        prior_ensemble,
        [True] * prior_ensemble.ensemble_size,
        0,
        None,
        "",
        "path_%",
        "name",
    )
    config = snake_oil_field_example.ert_config
    sample_prior(prior_ensemble, range(prior_ensemble.ensemble_size))
    create_run_path(run_context, config.substitution_list, config)

    # replace field file with symlink
    linkpath = f"{run_context[0].runpath}/permx.grdecl"
    targetpath = f"{run_context[0].runpath}/permx.grdecl.target"
    with open(targetpath, "a", encoding="utf-8"):
        pass
    os.remove(linkpath)
    os.symlink(targetpath, linkpath)

    # recreate directory structure
    create_run_path(run_context, config.substitution_list, config)

    # ensure field symlink is replaced by file
    assert not os.path.islink(linkpath)


def test_repr(minimum_case):
    assert repr(minimum_case).startswith("EnKFMain(size: 10, config")


@pytest.mark.usefixtures("use_tmpdir")
def test_ert_context():
    # Write a minimal config file with DEFINE
    with open("config_file.ert", "w", encoding="utf-8") as fout:
        fout.write("NUM_REALIZATIONS 1\nDEFINE MY_PATH <CONFIG_PATH>")
    ert_config = ErtConfig.from_file("config_file.ert")
    ert = EnKFMain(ert_config)
    context = ert.get_context()
    my_path = context["MY_PATH"]
    assert my_path == os.getcwd()
