from pathlib import Path
from unittest.mock import MagicMock

import pytest

import ert.run_models.everest_run_model
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.simulator.everest_to_ert import _everest_to_ert_config_dict

CONFIG_FILE = "config_minimal.yml"


@pytest.mark.integration_test
def test_seed(copy_math_func_test_data_to_tmp):
    random_seed = 42
    config = EverestConfig.load_file(CONFIG_FILE)
    config.environment.random_seed = random_seed

    run_model = EverestRunModel.create(config)
    assert random_seed == run_model._everest_config.environment.random_seed

    # Res
    ert_config = _everest_to_ert_config_dict(config)
    assert random_seed == ert_config["RANDOM_SEED"]


@pytest.mark.integration_test
def test_loglevel(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    config.environment.log_level = "info"
    run_model = EverestRunModel.create(config)
    config = run_model._everest_config
    assert len(EverestConfig.lint_config_dict(config.to_dict())) == 0


@pytest.mark.parametrize("iteration", [0, 1, 2])
def test_run_path(tmp_path, min_config, iteration, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ert.run_models.everest_run_model, "open_storage", MagicMock())

    model_realizations, control_indices = ([0, 2], [0, 1])
    config = EverestConfig(**min_config)
    run_model = EverestRunModel.create(config)
    ensemble_mock = MagicMock()
    ensemble_mock.iteration = iteration
    run_args = run_model._get_run_args(
        ensemble_mock, model_realizations, control_indices
    )
    assert [
        str(Path(run_arg.runpath).relative_to(Path().absolute()))
        for run_arg in run_args
    ] == [
        f"everest_output/simulation_folder/batch_{iteration}/geo_realization_0/simulation_0",
        f"everest_output/simulation_folder/batch_{iteration}/geo_realization_2/simulation_1",
    ]
