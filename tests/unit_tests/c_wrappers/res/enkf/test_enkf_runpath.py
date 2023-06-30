import fileinput
import os
from pathlib import Path

import pytest

from ert._c_wrappers.enkf.enkf_main import EnKFMain
from ert.config import ErtConfig


def test_with_gen_kw(copy_case, storage):
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
    main.sample_prior(prior.sim_fs, prior.active_realizations)
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
    main.sample_prior(prior.sim_fs, prior.active_realizations)
    main.createRunPath(prior)
    assert os.path.exists("storage/snake_oil/runpath/realization-0/iter-0")
    assert not os.path.exists(
        "storage/snake_oil/runpath/realization-0/iter-0/parameters.txt"
    )
    assert len(os.listdir("storage/snake_oil/runpath")) == 1
    assert len(os.listdir("storage/snake_oil/runpath/realization-0")) == 1
