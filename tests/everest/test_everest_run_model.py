from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import Mock, patch

import pytest

from ert.base_model_context import use_runtime_plugins
from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    QueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
    parse_string_to_bytes,
)
from ert.plugins import ErtRuntimePlugins, get_site_plugins
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from tests.everest.utils import everest_config_with_defaults


@pytest.fixture
def create_runmodel(min_config: dict, monkeypatch: pytest.MonkeyPatch) -> Callable:
    monkeypatch.setattr("ert.run_models.run_model.open_storage", Mock())
    monkeypatch.setattr(
        "ert.run_models.everest_run_model._get_internal_files", lambda _: {}
    )

    def _create_runmodel(
        simulator: dict[str, Any] | None = None,
        environment: dict[str, str] | None = None,
        config_path: str | None = None,
        config: dict | None = None,
    ) -> EverestRunModel:
        site_plugins = get_site_plugins()
        with use_runtime_plugins(site_plugins):
            return EverestRunModel.create(
                EverestConfig(
                    **(
                        min_config
                        | ({"simulator": simulator} if simulator else {})
                        | ({"environment": environment} if environment else {})
                        | (
                            {"config_path": config_path}
                            if config_path is not None
                            else {}
                        )
                        | (config if config is not None else {})
                    ),
                ),
                runtime_plugins=site_plugins,
            )

    return _create_runmodel


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@pytest.mark.parametrize("queue_system", ["lsf", "local", "torque", "slurm"])
def test_that_queue_system_name_passes_through_create(
    create_runmodel: Callable, queue_system: str
) -> None:
    runmodel = create_runmodel(simulator={"queue_system": {"name": queue_system}})
    assert runmodel.queue_config.queue_system == queue_system


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
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
        "activate_script": "foo_activate",
    }
    runmodel = create_runmodel(simulator={"queue_system": properties})
    for property_name, value in properties.items():
        assert getattr(runmodel.queue_config.queue_options, property_name) == value


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
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
    runmodel = create_runmodel(simulator={"queue_system": config})
    assert runmodel.queue_config.queue_options == config_class(**config)


def test_substitutions_from_everest_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, create_runmodel: Callable
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path("./hello/world/strong_optimizer.yml")
    config_dir = config_path.parent
    config_dir.mkdir(parents=True)
    config_path.touch()
    runmodel = create_runmodel(
        simulator={"queue_system": {"name": "lsf", "num_cpu": 1337}},
        config_path=str(config_path),
        environment={
            "simulation_folder": "the_simulations_dir",
            "output_folder": "custom_output_folder",
        },
    )

    assert runmodel.substitutions == {
        "<RUNPATH_FILE>": "hello/world/custom_output_folder/.res_runpath_list",
        "<RUNPATH>": (
            f"{config_dir}"
            "/custom_output_folder"
            "/the_simulations_dir/"
            "batch_<ITER>/realization_<REALIZATION_ID>/<SIM_DIR>"
        ),
        "<ECL_BASE>": "ECLBASE<IENS>",
        "<ECLBASE>": "ECLBASE<IENS>",
        "<NUM_CPU>": "1337",
        "<CONFIG_PATH>": str(config_dir),
        "<CONFIG_FILE>": "strong_optimizer",
    }


@pytest.mark.parametrize("random_seed", [None, 1234])
def test_that_random_seed_passes_through_create(
    create_runmodel: Callable, random_seed: int | None
) -> None:
    runmodel = create_runmodel(
        environment={"random_seed": random_seed} if random_seed is not None else None
    )

    if random_seed is None:
        assert runmodel.random_seed > 0
    else:
        assert runmodel.random_seed == random_seed


def test_cores_per_node_is_used_over_defaulted_num_cpu(
    create_runmodel: Callable,
) -> None:
    with pytest.warns(UserWarning, match="Ignoring cores_per_node as num_cpu was set"):
        runmodel = create_runmodel(config={"simulator": {"cores_per_node": 88}})
    assert runmodel.queue_config.queue_options.num_cpu == 88


def test_cores_per_node_is_ignored_num_cpu_is_set(
    create_runmodel: Callable,
) -> None:
    with pytest.warns(UserWarning, match="Ignoring cores_per_node.*"):
        runmodel = create_runmodel(
            config={
                "simulator": {
                    "cores_per_node": 88,
                    "queue_system": {"name": "lsf", "num_cpu": 99},
                }
            },
        )
    assert runmodel.queue_config.queue_options.num_cpu == 99


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@pytest.mark.parametrize(
    "config, config_class",
    [
        [
            {
                "name": "local",
                "max_running": 0,
                "submit_sleep": 0.0,
                "project_code": "",
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
                "exclude_host": "",
                "lsf_queue": "lsf_queue",
                "lsf_resource": "",
            },
            LsfQueueOptions,
        ],
    ],
)
def test_everest_to_ert_queue_config(config, config_class, create_runmodel):
    """Note that these objects are used directly in the Everest
    config, and if you have to make changes to this test, it is likely
    that it is a breaking change to Everest"""
    general_queue_options = {"max_running": 10}
    general_options = {"resubmit_limit": 7}

    runmodel = create_runmodel(
        simulator={"queue_system": config | general_queue_options} | general_options,
    )

    assert runmodel.queue_config.queue_options == config_class(
        **(config | general_queue_options)
    )


@pytest.mark.usefixtures("use_site_configurations_with_lsf_queue_options")
def test_that_site_config_queue_options_do_not_override_user_queue_config(
    min_config, change_to_tmpdir
):
    ever_config = everest_config_with_defaults(
        simulator={"queue_system": {"name": "local"}}, model={"realizations": [0]}
    )

    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        config = EverestRunModel.create(
            ever_config, "some_exp_name", "batch", runtime_plugins=site_plugins
        )
        assert config.queue_system == "local"


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_queue_settings_are_taken_from_site_config(
    create_runmodel, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    site_plugins = get_site_plugins()
    site_plugins.queue_options = LsfQueueOptions(
        name="lsf", lsf_resource="my_resource", lsf_queue="my_queue"
    )

    with mock.patch(
        "ert.plugins.plugin_manager.ErtRuntimePlugins",
        return_value=site_plugins,
    ):
        runmodel = create_runmodel()

        assert runmodel.queue_config.queue_options == LsfQueueOptions(
            lsf_queue="my_queue", lsf_resource="my_resource"
        )


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@pytest.mark.parametrize(
    "max_memory",
    [0, 1, "0", "1", "1b", "1k", "1m", "1g", "1t", "1p", "1G", "1 G", "1Gb", "1 Gb"],
)
def test_that_max_memory_is_passed_to_ert_unchanged(
    create_runmodel, max_memory
) -> None:
    runmodel = create_runmodel(simulator={"max_memory": max_memory})

    assert runmodel.queue_config.queue_options.realization_memory == (
        parse_string_to_bytes(max_memory) if isinstance(max_memory, str) else max_memory
    )


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_max_memory_none_is_not_passed_to_ert(create_runmodel) -> None:
    runmodel = create_runmodel()
    assert runmodel.queue_config.queue_options.realization_memory == 0


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_resubmit_limit_is_set(create_runmodel) -> None:
    runmodel = create_runmodel(simulator={"resubmit_limit": 2})
    assert runmodel.queue_config.max_submit == 3


def test_that_general_user_queue_options_overrides_site_queue_options_via_runmodel(
    min_config,
):
    local_queue_options = LocalQueueOptions(realization_memory="1110Gb", num_cpu=10)

    def ErtRuntimePluginsWithCustomQueueOptions(**kwargs):
        return ErtRuntimePlugins(**(kwargs | {"queue_options": local_queue_options}))

    with (
        patch(
            "ert.plugins.plugin_manager.ErtRuntimePlugins",
            ErtRuntimePluginsWithCustomQueueOptions,
        ),
    ):
        site_plugins = get_site_plugins()

    with (
        use_runtime_plugins(site_plugins),
        pytest.warns(UserWarning, match="Ignoring cores_per_node as num_cpu was set"),
    ):
        ever_config = EverestConfig(
            **(
                min_config
                | {
                    "simulator": {
                        "cores_per_node": 2,
                        "max_memory": "2Gb",
                    }
                }
            )
        )
        run_model = EverestRunModel.create(
            everest_config=ever_config, runtime_plugins=site_plugins
        )
        assert (
            run_model.queue_config.queue_options.realization_memory
            == parse_string_to_bytes("2Gb")
        )
        assert run_model.queue_config.queue_options.num_cpu == 2
