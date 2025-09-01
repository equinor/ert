from pathlib import Path
from unittest.mock import MagicMock

import pytest

import ert.run_models.everest_run_model
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.simulator.everest_to_ert import _everest_to_ert_config_dict


def test_that_seed_in_everestconfig_is_passed_to_ert_config(change_to_tmpdir):
    random_seed = 42
    config = EverestConfig.with_defaults()
    config.environment.random_seed = random_seed
    ert_config = _everest_to_ert_config_dict(config)
    assert ert_config["RANDOM_SEED"] == random_seed


def test_that_default_everestconfig_lints():
    config = EverestConfig.with_defaults()
    config.environment.log_level = "info"
    assert len(EverestConfig.lint_config_dict(config.to_dict())) == 0


@pytest.mark.parametrize("iteration", [0, 1, 2])
def test_that_runpath_strings_are_generated_correctly(
    tmp_path, min_config, iteration, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ert.run_models.run_model, "open_storage", MagicMock())

    model_realizations = [0, 2]
    config = EverestConfig(**min_config)
    run_model = EverestRunModel.create(config)
    ensemble_mock = MagicMock()
    ensemble_mock.iteration = iteration
    run_args = run_model._get_run_args(ensemble_mock, model_realizations)
    assert [
        str(Path(run_arg.runpath).relative_to(Path().absolute()))
        for run_arg in run_args
    ] == [
        f"everest_output/simulation_folder/batch_{iteration}/geo_realization_0/simulation_0",
        f"everest_output/simulation_folder/batch_{iteration}/geo_realization_2/simulation_1",
    ]
