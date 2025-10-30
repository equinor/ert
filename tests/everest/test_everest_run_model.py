from collections.abc import Callable
from unittest.mock import Mock

import pytest

from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    QueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
)
from ert.plugins import ErtPluginContext
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig


@pytest.fixture
def create_runmodel(min_config: dict, monkeypatch: pytest.MonkeyPatch) -> Callable:
    monkeypatch.setattr("ert.run_models.run_model.open_storage", Mock())

    def _create_runmodel(
        queue_system: dict[str, str | int | bool | float],
    ) -> EverestRunModel:
        with ErtPluginContext() as runtime_plugins:
            runtime_plugins.queue_options = None
            return EverestRunModel.create(
                EverestConfig(
                    **(min_config | {"simulator": {"queue_system": queue_system}})
                ),
                runtime_plugins=runtime_plugins,
            )

    return _create_runmodel


@pytest.mark.parametrize("queue_system", ["lsf", "local", "torque", "slurm"])
def test_that_queue_system_name_passes_through_create(
    create_runmodel: Callable, queue_system: str
) -> None:
    runmodel = create_runmodel({"name": queue_system})
    assert runmodel.queue_config.queue_system == queue_system


def test_general_queue_options_properties_pass_through_create(
    create_runmodel: Callable,
) -> None:
    properties = {
        "name": "lsf",
        "max_running": 11,
        "submit_sleep": 22,
        "num_cpu": 33,
        "realization_memory": 44,
        "project_code": "foo_code",
        "job_script": "foo_script",
        "activate_script": "foo_activate",
    }
    runmodel = create_runmodel(properties)
    for property_name, value in properties.items():
        assert getattr(runmodel.queue_config.queue_options, property_name) == value


@pytest.mark.parametrize(
    "config, config_class",
    [
        [
            {
                "name": "local",
                "max_running": 0,
                "submit_sleep": 0.0,
                "project_code": "foo",
                "activate_script": "activate_script",
            },
            LocalQueueOptions,
        ],
        [
            {
                "name": "torque",
                "qsub_cmd": "qsub",
                "qstat_cmd": "qstat",
                "qdel_cmd": "qdel",
                "queue": "queue",
                "cluster_label": "cluster_label",
                "job_prefix": "job_prefix",
                "keep_qsub_output": False,
            },
            TorqueQueueOptions,
        ],
        [
            {
                "name": "slurm",
                "sbatch": "sbatch",
                "scancel": "scancel",
                "scontrol": "scontrol",
                "sacct": "sacct",
                "squeue": "squeue",
                "exclude_host": "exclude_host",
                "include_host": "include_host",
                "partition": "some_partition",
                "squeue_timeout": 2.0,
                "max_runtime": 10,
            },
            SlurmQueueOptions,
        ],
        [
            {
                "name": "lsf",
                "bhist_cmd": "bhist",
                "bjobs_cmd": "bjobs",
                "bkill_cmd": "bkill",
                "bsub_cmd": "bsub",
                "exclude_host": "hosts",
                "lsf_queue": "lsf_queue",
                "lsf_resource": "some_resource",
            },
            LsfQueueOptions,
        ],
    ],
)
def test_queue_options_properties_pass_through_create(
    create_runmodel: Callable,
    config: dict[str, str | int | float | bool],
    config_class: QueueOptions,
) -> None:
    runmodel = create_runmodel(config)
    assert runmodel.queue_config.queue_options == config_class(**config)
