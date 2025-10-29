from pathlib import Path
from unittest.mock import MagicMock

import pytest

import ert.run_models.everest_run_model
from ert.plugins import ErtPluginContext
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig


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
    perturbations = [-1, -1]
    config = EverestConfig(**min_config)

    with ErtPluginContext() as runtime_plugins:
        run_model = EverestRunModel.create(config, runtime_plugins=runtime_plugins)

    ensemble_mock = MagicMock()
    ensemble_mock.iteration = iteration
    run_args = run_model._get_run_args(ensemble_mock, model_realizations, perturbations)
    assert [
        str(Path(run_arg.runpath).relative_to(Path().absolute()))
        for run_arg in run_args
    ] == [
        f"everest_output/simulation_folder/batch_{iteration}/realization_0/evaluation_0",
        f"everest_output/simulation_folder/batch_{iteration}/realization_2/evaluation_0",
    ]
