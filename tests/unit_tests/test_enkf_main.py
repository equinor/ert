import os
from pathlib import Path
from textwrap import dedent

import pytest
from ecl.summary import EclSum

from ert.config import AnalysisConfig, EnsembleConfig, ErtConfig, ModelConfig
from ert.enkf_main import EnKFMain


@pytest.mark.unstable
def test_ecl_config_creation(minimum_case):
    assert isinstance(minimum_case.analysisConfig(), AnalysisConfig)
    assert isinstance(minimum_case.ensembleConfig(), EnsembleConfig)

    with pytest.raises(AssertionError):  # Null pointer!
        assert isinstance(minimum_case.ensembleConfig().refcase, EclSum)


@pytest.fixture(name="enkf_main")
def enkf_main_fixture(tmp_path, monkeypatch):
    (tmp_path / "test.ert").write_text("NUM_REALIZATIONS 1\nJOBNAME name%d")
    monkeypatch.chdir(tmp_path)
    yield EnKFMain(ErtConfig.from_file("test.ert"))


@pytest.mark.parametrize(
    "config_dict",
    [
        {"ECLBASE": "name<IENS>"},
        {"JOBNAME": "name<IENS>"},
    ],
)
def test_create_run_context(monkeypatch, enkf_main, prior_ensemble, config_dict):
    iteration = 0
    ensemble_size = 10
    enkf_main = EnKFMain(ErtConfig.from_dict(config_dict))

    run_context = enkf_main.ensemble_context(
        prior_ensemble, [True] * ensemble_size, iteration=iteration
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

    substitutions = enkf_main.get_context()
    assert "<RUNPATH>" in substitutions
    assert substitutions.get("<ECL_BASE>") == "name<IENS>"
    assert substitutions.get("<ECLBASE>") == "name<IENS>"


def test_create_run_context_separate_base_and_name(
    monkeypatch, enkf_main, prior_ensemble
):
    iteration = 0
    ensemble_size = 10
    enkf_main = EnKFMain(
        ErtConfig.from_dict({"JOBNAME": "name<IENS>", "ECLBASE": "base<IENS>"})
    )

    run_context = enkf_main.ensemble_context(
        prior_ensemble, [True] * ensemble_size, iteration=iteration
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

    substitutions = enkf_main.get_context()
    assert "<RUNPATH>" in substitutions
    assert substitutions.get("<ECL_BASE>") == "base<IENS>"
    assert substitutions.get("<ECLBASE>") == "base<IENS>"


def test_assert_symlink_deleted(snake_oil_field_example, storage):
    ert = snake_oil_field_example
    experiment_id = storage.create_experiment(
        parameters=ert.ensembleConfig().parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=100
    )

    # create directory structure
    run_context = ert.ensemble_context(
        prior_ensemble, [True] * (ert.getEnsembleSize()), iteration=0
    )
    ert.sample_prior(run_context.sim_fs, run_context.active_realizations)
    ert.createRunPath(run_context)

    # replace field file with symlink
    linkpath = f"{run_context[0].runpath}/permx.grdecl"
    targetpath = f"{run_context[0].runpath}/permx.grdecl.target"
    with open(targetpath, "a", encoding="utf-8"):
        pass
    os.remove(linkpath)
    os.symlink(targetpath, linkpath)

    # recreate directory structure
    ert.createRunPath(run_context)

    # ensure field symlink is replaced by file
    assert not os.path.islink(linkpath)


def test_repr(minimum_case):
    assert repr(minimum_case).startswith("EnKFMain(size: 10, config")


def test_bootstrap(minimum_case):
    assert bool(minimum_case)


def test_config(minimum_case):
    assert isinstance(minimum_case.ensembleConfig(), EnsembleConfig)
    assert isinstance(minimum_case.analysisConfig(), AnalysisConfig)
    assert isinstance(minimum_case.getModelConfig(), ModelConfig)


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
