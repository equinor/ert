import itertools
import stat
from pathlib import Path
from shutil import which
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

import everest
from ert.config import ConfigWarning
from ert.config.ensemble_config import EnsembleConfig
from ert.config.ert_config import create_and_hook_workflows, workflows_from_dict
from ert.config.model_config import ModelConfig
from ert.config.parsing import ConfigKeys as ErtConfigKeys
from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    QueueConfig,
    SlurmQueueOptions,
    TorqueQueueOptions,
)
from everest.config import EverestConfig
from everest.simulator.everest_to_ert import (
    _everest_to_ert_config_dict,
    _get_installed_forward_model_steps,
    everest_to_ert_config_dict,
    get_forward_model_steps,
    get_substitutions,
    get_workflow_jobs,
)
from tests.everest.utils import skipif_no_everest_models


@pytest.mark.usefixtures("no_plugins")
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
def test_everest_to_ert_queue_config(config, config_class, tmp_path, monkeypatch):
    """Note that these objects are used directly in the Everest
    config, and if you have to make changes to this test, it is likely
    that it is a breaking change to Everest"""
    monkeypatch.chdir(tmp_path)
    general_queue_options = {"max_running": 10}
    general_options = {"resubmit_limit": 7}
    config |= general_queue_options
    ever_config = EverestConfig.with_defaults(
        simulator={"queue_system": config} | general_options,
        model={"realizations": [0]},
    )

    config_dict = everest_to_ert_config_dict(ever_config)
    queue_config = QueueConfig.from_dict(config_dict)
    queue_config.queue_options = ever_config.simulator.queue_system
    queue_config.queue_system = ever_config.simulator.queue_system.name

    assert queue_config.queue_options == config_class(**config)


def test_default_installed_jobs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    jobs = [
        "render",
        "make_directory",
        "copy_directory",
        "copy_file",
        "move_file",
        "symlink",
    ]
    ever_config = EverestConfig.with_defaults(
        **yaml.safe_load(
            dedent(f"""
    model: {{"realizations": [0]}}
    forward_model: {jobs}
    """)
        )
    )

    config_dict = everest_to_ert_config_dict(ever_config)
    substitutions = get_substitutions(
        config_dict=config_dict,
        model_config=ModelConfig(),
        runpath_file=MagicMock(),
        num_cpu=0,
    )
    forward_model_steps, _ = get_forward_model_steps(
        ever_config, config_dict, substitutions
    )

    # Index 0 is the copy job for wells.json
    assert [c.name for c in forward_model_steps[1:]] == jobs


@pytest.mark.filterwarnings(
    "ignore:Config contains a SUMMARY key but no forward model steps"
)
def test_combined_wells_everest_to_ert(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("my_file").touch()
    Path("my_executable").touch(mode=stat.S_IEXEC)
    ever_config = EverestConfig.with_defaults(
        **yaml.safe_load(
            dedent("""
    model: {"realizations": [0]}
    wells: [{ name: fakename}]
    definitions: {eclbase: my_test_case}
    install_jobs:
      - name: nothing
        executable: my_executable
    forward_model:
      - job: nothing
        results:
          file_name: something
          type: summary
    """)
        )
    )
    config_dict = everest_to_ert_config_dict(ever_config)
    ensemble_config = EnsembleConfig.from_dict(config_dict)

    assert "WOPR:fakename" in ensemble_config.response_configs["summary"].keys


@pytest.mark.parametrize(
    "source, target, symlink, cmd",
    [
        ["source_file", "target_file", True, "symlink"],
        ["source_file", "target_file", False, "copy_file"],
        ["source_folder", "target_folder", False, "copy_directory"],
    ],
)
def test_install_data_no_init(tmp_path, source, target, symlink, cmd, monkeypatch):
    """
    Configure the everest config with the install_data section and check that the
    correct ert forward models are created
    """
    monkeypatch.chdir(tmp_path)
    Path("source_file").touch()
    Path.mkdir("source_folder")
    ever_config = EverestConfig.with_defaults(
        model={"realizations": [0]},
        install_data=[{"source": source, "target": target, "link": symlink}],
    )

    errors = EverestConfig.lint_config_dict(ever_config.to_dict())
    assert len(errors) == 0

    config_dict = everest_to_ert_config_dict(ever_config)
    substitutions = get_substitutions(
        config_dict=config_dict,
        model_config=ModelConfig(),
        runpath_file=MagicMock(),
        num_cpu=0,
    )
    forward_model_steps, _ = get_forward_model_steps(
        ever_config, config_dict, substitutions
    )

    expected_fm = next(val for val in forward_model_steps if val.name == cmd)
    assert expected_fm.arglist == [f"./{source}", target]


@pytest.mark.integration_test
@skipif_no_everest_models
@pytest.mark.everest_models_test
@pytest.mark.skip_mac_ci
def test_summary_default_no_opm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    everconf = EverestConfig.with_defaults(
        forward_model=[
            {
                "job": "eclipse100 eclipse/model/EgG.DATA --version 2020.2",
                "results": {
                    "file_name": "eclipse/model/EGG",
                    "type": "summary",
                    "keys": ["*"],
                },
            }
        ]
    )
    # Read wells from the config instead of using opm
    wells = [w.name for w in everconf.wells]
    sum_keys = (
        list(everest.simulator.DEFAULT_DATA_SUMMARY_KEYS)
        + list(everest.simulator.DEFAULT_FIELD_SUMMARY_KEYS)
        + [
            f"{k}:{w}"
            for k, w in itertools.product(
                everest.simulator.DEFAULT_WELL_SUMMARY_KEYS, wells
            )
        ]
    )
    sum_keys = [list(set(sum_keys))]
    res_conf = _everest_to_ert_config_dict(everconf)

    assert set(sum_keys[0]) == set(res_conf[ErtConfigKeys.SUMMARY][0]) - {"*"}


def test_workflow_job_deprecated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("TEST").write_text("EXECUTABLE echo", encoding="utf-8")
    workflow_jobs = [{"name": "test", "source": "TEST"}]
    with pytest.warns(
        ConfigWarning, match="`install_workflow_jobs: source` is deprecated"
    ):
        ever_config = EverestConfig.with_defaults(
            install_workflow_jobs=workflow_jobs, model={"realizations": [0]}
        )
    config_dict = everest_to_ert_config_dict(ever_config)
    substitutions = get_substitutions(
        config_dict=config_dict,
        model_config=ModelConfig(),
        runpath_file=MagicMock(),
        num_cpu=0,
    )
    workflow_jobs, _, _ = workflows_from_dict(config_dict, substitutions)

    jobs = workflow_jobs.get("test")
    assert jobs.executable == "echo"


def test_workflow_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workflow_jobs = [{"name": "test", "executable": which("echo")}]
    ever_config = EverestConfig.with_defaults(
        install_workflow_jobs=workflow_jobs, model={"realizations": [0]}
    )
    workflow_jobs = get_workflow_jobs(ever_config)
    jobs = workflow_jobs.get("test")
    assert jobs.executable == which("echo")


def test_workflow_job_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    echo = which("echo")
    workflow_jobs = [
        {"name": "test", "executable": which("true")},
        {"name": "test", "executable": echo},
    ]
    ever_config = EverestConfig.with_defaults(
        install_workflow_jobs=workflow_jobs, model={"realizations": [0]}
    )
    with pytest.warns(
        ConfigWarning,
        match=f"Duplicate workflow job with name 'test', overriding it with {echo!r}.",
    ):
        workflow_jobs = get_workflow_jobs(ever_config)
    jobs = workflow_jobs.get("test")
    assert jobs.executable == echo


def test_workflows_deprecated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("TEST").write_text("EXECUTABLE echo", encoding="utf-8")
    workflow_jobs = [{"name": "my_test", "source": "TEST"}]
    workflow = {"pre_simulation": ["my_test"]}
    with pytest.warns(
        ConfigWarning, match="`install_workflow_jobs: source` is deprecated"
    ):
        ever_config = EverestConfig.with_defaults(
            workflows=workflow,
            model={"realizations": [0]},
            install_workflow_jobs=workflow_jobs,
        )
    config_dict = everest_to_ert_config_dict(ever_config)
    substitutions = get_substitutions(
        config_dict=config_dict,
        model_config=ModelConfig(),
        runpath_file=MagicMock(),
        num_cpu=0,
    )
    _, workflows, _ = workflows_from_dict(config_dict, substitutions)

    jobs = workflows.get("pre_simulation")
    assert jobs.cmd_list[0][0].name == "my_test"
    assert jobs.cmd_list[0][0].executable == "echo"


def test_workflows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workflow_jobs = [{"name": "my_test", "executable": which("echo")}]
    workflow = {"pre_simulation": ["my_test"]}
    ever_config = EverestConfig.with_defaults(
        workflows=workflow,
        model={"realizations": [0]},
        install_workflow_jobs=workflow_jobs,
    )
    config_dict = everest_to_ert_config_dict(ever_config)
    substitutions = get_substitutions(
        config_dict=config_dict,
        model_config=ModelConfig(),
        runpath_file=MagicMock(),
        num_cpu=0,
    )
    workflow_jobs = get_workflow_jobs(ever_config)
    workflows, _ = create_and_hook_workflows(config_dict, workflow_jobs, substitutions)
    jobs = workflows.get("pre_simulation")
    assert jobs.cmd_list[0][0].name == "my_test"
    assert jobs.cmd_list[0][0].executable == which("echo")


def test_user_config_jobs_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing_job = "copy_file"
    ever_config = EverestConfig.with_defaults(model={"realizations": [0]})
    config_dict = everest_to_ert_config_dict(ever_config)
    installed_forward_model_steps = _get_installed_forward_model_steps(
        ever_config, config_dict
    )

    assert existing_job in installed_forward_model_steps

    echo = which("echo")
    ever_config_new = EverestConfig.with_defaults(
        model={"realizations": [0]},
        install_jobs=[{"name": existing_job, "executable": echo}],
    )
    config_dict_new = everest_to_ert_config_dict(ever_config_new)
    with pytest.warns(
        ConfigWarning,
        match=(
            f"Duplicate forward model with name '{existing_job}'"
            f", overriding it with '{echo}'."
        ),
    ):
        installed_forward_model_steps_new = _get_installed_forward_model_steps(
            ever_config_new, config_dict_new
        )

    assert installed_forward_model_steps_new.get(existing_job).executable == echo


@pytest.mark.usefixtures("no_plugins")
def test_that_queue_settings_are_taken_from_site_config(
    min_config, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    assert "simulator" not in min_config  # Double check
    Path("site-config").write_text(
        dedent("""
    QUEUE_SYSTEM LSF
    QUEUE_OPTION LSF LSF_RESOURCE my_resource
    QUEUE_OPTION LSF LSF_QUEUE my_queue
    """),
        encoding="utf-8",
    )
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(min_config, f)

    with patch(
        "ert.config.ert_config.site_config_location", return_value="site-config"
    ):
        config = EverestConfig.load_file("config.yml")
        assert config.simulator.queue_system == LsfQueueOptions(
            lsf_queue="my_queue", lsf_resource="my_resource"
        )
        queue_config = QueueConfig.from_dict(everest_to_ert_config_dict(config))
        assert queue_config.queue_options == LsfQueueOptions(
            lsf_queue="my_queue", lsf_resource="my_resource"
        )


def test_passthrough_explicit_summary_keys(change_to_tmpdir):
    custom_sum_keys = [
        "GOIR:PRODUC",
        "GOIT:INJECT",
        "GOIT:PRODUC",
        "GWPR:INJECT",
        "GWPR:PRODUC",
        "GWPT:INJECT",
        "GWPT:PRODUC",
        "GWIR:INJECT",
    ]

    config = EverestConfig.with_defaults(
        forward_model=[
            {
                "job": "eclipse100 eclipse/model/EgG.DATA --version 2020.2",
                "results": {
                    "file_name": "eclipse/model/EGG",
                    "type": "summary",
                    "keys": custom_sum_keys,
                },
            }
        ]
    )

    ert_config = _everest_to_ert_config_dict(config)
    assert set(custom_sum_keys).issubset(set(ert_config[ErtConfigKeys.SUMMARY][0]))


@pytest.mark.usefixtures("no_plugins")
@pytest.mark.parametrize(
    "max_memory",
    [
        None,
        0,
        1,
        "0",
        "1",
        "1b",
        "1k",
        "1m",
        "1g",
        "1t",
        "1p",
        "1G",
        "1 G",
        "1Gb",
        "1 Gb",
    ],
)
def test_that_max_memory_is_valid(max_memory) -> None:
    EverestConfig.with_defaults(simulator={"max_memory": max_memory})


@pytest.mark.usefixtures("no_plugins")
@pytest.mark.parametrize(
    "max_memory",
    [-1, "-1", "-1G", "-1 G", "-1Gb"],
)
def test_that_negative_max_memory_fails(max_memory) -> None:
    with pytest.raises(
        ValidationError, match=f"Negative memory does not make sense in {max_memory}"
    ):
        EverestConfig.with_defaults(simulator={"max_memory": max_memory})


@pytest.mark.usefixtures("no_plugins")
@pytest.mark.parametrize(
    "max_memory",
    ["1x", "1 x", "1 xy", "foo"],
)
def test_that_invalid_max_memory_fails(max_memory) -> None:
    with pytest.raises(
        ValidationError, match=f"Could not understand byte unit in {max_memory}"
    ):
        EverestConfig.with_defaults(simulator={"max_memory": max_memory})


@pytest.mark.usefixtures("no_plugins")
@pytest.mark.parametrize(
    "max_memory",
    [0, 1, "0", "1", "1b", "1k", "1m", "1g", "1t", "1p", "1G", "1 G", "1Gb", "1 Gb"],
)
def test_that_max_memory_is_passed_to_ert_unchanged(
    change_to_tmpdir, max_memory
) -> None:
    ever_config = EverestConfig.with_defaults(simulator={"max_memory": max_memory})
    config_dict = everest_to_ert_config_dict(ever_config)
    assert config_dict[ErtConfigKeys.REALIZATION_MEMORY] == str(max_memory)


@pytest.mark.usefixtures("no_plugins")
def test_that_max_memory_none_is_not_passed_to_ert(change_to_tmpdir) -> None:
    ever_config = EverestConfig.with_defaults()
    config_dict = everest_to_ert_config_dict(ever_config)
    assert ErtConfigKeys.REALIZATION_MEMORY not in config_dict


@pytest.mark.usefixtures("no_plugins")
def test_that_resubmit_limit_is_set(change_to_tmpdir) -> None:
    ever_config = EverestConfig.with_defaults(simulator={"resubmit_limit": 0})
    config_dict = everest_to_ert_config_dict(ever_config)
    assert config_dict[ErtConfigKeys.MAX_SUBMIT] == 1


@pytest.mark.parametrize(
    "max_memory, realization_memory", [(None, 0), (999, 999), ("1Gb", 1073741824)]
)
def test_that_max_memory_is_passed_to_realization_memory(
    change_to_tmpdir, max_memory, realization_memory
) -> None:
    config = EverestConfig.with_defaults(simulator={"max_memory": max_memory})
    assert config.simulator.queue_system.realization_memory == realization_memory
