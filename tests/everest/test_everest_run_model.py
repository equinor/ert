from unittest.mock import Mock

import pytest

from ert.plugins import ErtPluginContext
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig

minimal_config = {
    "controls": [
        {
            "name": "control1",
            "type": "generic_control",
            "min": 0,
            "max": 10,
            "perturbation_magnitude": 0.01,
            "variables": [
                {"name": "w", "initial_guess": 1},
            ],
        }
    ],
    "objective_functions": [{"name": "some_function"}],
    "model": {"realizations": [0]},
    "config_path": ".",
}


@pytest.mark.parametrize("queue_system", ["lsf", "local", "torque", "slurm"])
def test_that_queue_system_name_passes_through_create(
    monkeypatch: pytest.MonkeyPatch, queue_system: str
) -> None:
    monkeypatch.setattr("ert.run_models.run_model.open_storage", Mock())
    with ErtPluginContext() as runtime_plugins:
        runmodel = EverestRunModel.create(
            EverestConfig(
                **(
                    minimal_config
                    | {
                        "simulator": {
                            "queue_system": {"name": queue_system},
                        }
                    }
                )
            ),
            runtime_plugins=runtime_plugins,
        )
    assert runmodel.queue_config.queue_system == queue_system


def test_that_num_cpu_passes_through_create(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ert.run_models.run_model.open_storage", Mock())
    with ErtPluginContext() as runtime_plugins:
        runmodel = EverestRunModel.create(
            EverestConfig(
                **(
                    minimal_config
                    | {
                        "simulator": {
                            "cores_per_node": 4,
                            "queue_system": {"name": "local"},
                        }
                    }
                )
            ),
            runtime_plugins=runtime_plugins,
        )
    assert runmodel.queue_config.queue_options.num_cpu == 4


def test_that_max_running_passes_through_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ert.run_models.run_model.open_storage", Mock())
    with ErtPluginContext() as runtime_plugins:
        runmodel = EverestRunModel.create(
            EverestConfig(
                **(
                    minimal_config
                    | {
                        "simulator": {
                            "queue_system": {
                                "name": "local",
                                "max_running": 4,
                            },
                        }
                    }
                )
            ),
            runtime_plugins=runtime_plugins,
        )
    assert runmodel.queue_config.queue_options.max_running == 4


def test_that_submit_sleep_passes_through_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ert.run_models.run_model.open_storage", Mock())
    with ErtPluginContext() as runtime_plugins:
        runmodel = EverestRunModel.create(
            EverestConfig(
                **(
                    minimal_config
                    | {
                        "simulator": {
                            "queue_system": {
                                "name": "local",
                                "submit_sleep": 8,
                            },
                        }
                    }
                )
            ),
            runtime_plugins=runtime_plugins,
        )
    assert runmodel.queue_config.queue_options.submit_sleep == 8


def test_that_realization_memory_passes_through_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ert.run_models.run_model.open_storage", Mock())
    with ErtPluginContext() as runtime_plugins:
        runmodel = EverestRunModel.create(
            EverestConfig(
                **(
                    minimal_config
                    | {
                        "simulator": {
                            "queue_system": {
                                "name": "local",
                                "realization_memory": "12Gb",
                            },
                        }
                    }
                )
            ),
            runtime_plugins=runtime_plugins,
        )
    assert runmodel.queue_config.queue_options.realization_memory == 12 * 1024**3
