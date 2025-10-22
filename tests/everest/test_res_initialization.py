import itertools
import os
import stat
from pathlib import Path
from shutil import which
from textwrap import dedent
from unittest import mock

import pytest
import yaml
from pydantic import ValidationError

import everest
from ert.config import ConfigWarning, SummaryConfig
from ert.config.parsing import ConfigKeys as ErtConfigKeys
from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    QueueConfig,
    SlurmQueueOptions,
    TorqueQueueOptions,
)
from ert.plugins import ErtPluginContext, ErtRuntimePlugins
from ert.run_models.everest_run_model import EverestRunModel, _get_workflow_jobs
from everest.config import EverestConfig, InstallDataConfig
from everest.simulator.everest_to_ert import everest_to_ert_config_dict


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


@pytest.mark.usefixtures("use_site_configurations_with_lsf_queue_options")
def test_that_site_config_queue_options_do_not_override_user_queue_config(
    min_config, monkeypatch, change_to_tmpdir
):
    ever_config = EverestConfig.with_defaults(
        simulator={"queue_system": {"name": "local"}}, model={"realizations": [0]}
    )

    with ErtPluginContext() as runtime_plugins:
        config = EverestRunModel.create(
            ever_config, "some_exp_name", "batch", runtime_plugins=runtime_plugins
        )
        assert config.queue_system == "local"


def test_default_installed_jobs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    jobs = [
        "template_render",
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

    with ErtPluginContext() as runtime_plugins:
        runmodel = EverestRunModel.create(ever_config, runtime_plugins=runtime_plugins)

    assert [fm.name for fm in runmodel.forward_model_steps[1:]] == jobs


@pytest.mark.filterwarnings(
    "ignore:Config contains a SUMMARY key but no forward model steps"
)
@pytest.mark.parametrize(
    "config_yaml",
    [
        dedent("""
    wells: [{ name: fakename}]
    """),
        dedent("""
    controls:
      - name: default_group
        type: well_control
        initial_guess: 0.5
        perturbation_magnitude: 0.01
        variables:
          - name: fakename
            min: 0
            max: 1
    """),
    ],
)
def test_combined_wells_everest_to_ert(tmp_path, monkeypatch, config_yaml):
    monkeypatch.chdir(tmp_path)
    Path("my_file").touch()
    Path("my_executable").touch(mode=stat.S_IEXEC)
    ever_config = EverestConfig.with_defaults(
        **yaml.safe_load(
            config_yaml
            + dedent("""
    model: {"realizations": [0]}
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

    with ErtPluginContext() as runtime_plugins:
        runmodel = EverestRunModel.create(
            ever_config, "some_exp_name", "batch", runtime_plugins=runtime_plugins
        )
    smry_config = next(
        r for r in runmodel.response_configuration if isinstance(r, SummaryConfig)
    )
    assert "WOPR:fakename" in smry_config.keys


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

    with ErtPluginContext() as runtime_plugins:
        runmodel = EverestRunModel.create(ever_config, runtime_plugins=runtime_plugins)

    matching_fm_step = next(fm for fm in runmodel.forward_model_steps if fm.name == cmd)
    assert matching_fm_step.arglist == [f"./{source}", target]


@pytest.mark.integration_test
@pytest.mark.skip_mac_ci
@pytest.mark.parametrize("wells_config", [None, [{"name": "default_name"}]])
def test_summary_default_no_opm(tmp_path, monkeypatch, wells_config):
    monkeypatch.chdir(tmp_path)
    everconf = EverestConfig.with_defaults(
        wells=wells_config,
        controls=[
            {
                "name": "default_group",
                "type": "well_control",
                "initial_guess": 0.5,
                "perturbation_magnitude": 0.01,
                "variables": [
                    {"name": "default_name", "min": 0, "max": 1},
                ],
            }
        ],
        forward_model=[
            {
                "job": "eclipse100 eclipse/model/EgG.DATA --version 2020.2",
                "results": {
                    "file_name": "eclipse/model/EGG",
                    "type": "summary",
                    "keys": ["*"],
                },
            }
        ],
    )
    # Read wells from the config instead of using opm
    wells = (
        [
            variable.name
            for control in everconf.controls
            for variable in control.variables
            if control.type == "well_control"
        ]
        if wells_config is None
        else [w.name for w in everconf.wells]
    )
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

    with ErtPluginContext() as runtime_plugins:
        runmodel = EverestRunModel.create(
            everconf, "some_exp_name", "batch", runtime_plugins=runtime_plugins
        )
    smry_config = next(
        r for r in runmodel.response_configuration if isinstance(r, SummaryConfig)
    )

    assert set(sum_keys[0]) == set(smry_config.keys) - {"*"}


def test_workflow_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workflow_jobs = [{"name": "test", "executable": which("echo")}]
    ever_config = EverestConfig.with_defaults(
        install_workflow_jobs=workflow_jobs, model={"realizations": [0]}
    )
    workflow_jobs = _get_workflow_jobs(ever_config)
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
        workflow_jobs = _get_workflow_jobs(ever_config)
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
    with ErtPluginContext() as runtime_plugins:
        runmodel = EverestRunModel.create(ever_config, runtime_plugins=runtime_plugins)
        assert (
            runmodel.hooked_workflows.popitem()[1][0].cmd_list[0][0].executable
            == "echo"
        )


def test_workflows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    executable = which("echo")
    workflow_jobs = [{"name": "my_test", "executable": executable}]
    workflow = {"pre_simulation": ["my_test"]}
    ever_config = EverestConfig.with_defaults(
        workflows=workflow,
        model={"realizations": [0]},
        install_workflow_jobs=workflow_jobs,
    )
    with ErtPluginContext() as runtime_plugins:
        runmodel = EverestRunModel.create(ever_config, runtime_plugins=runtime_plugins)
        assert (
            runmodel.hooked_workflows.popitem()[1][0].cmd_list[0][0].executable
            == executable
        )


def test_user_config_jobs_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing_job = "copy_file"
    ever_config = EverestConfig.with_defaults(model={"realizations": [0]})
    with ErtPluginContext() as runtime_plugins:
        runmodel = EverestRunModel.create(ever_config, runtime_plugins=runtime_plugins)

    assert runmodel.forward_model_steps[0].name == existing_job
    runmodel._storage.close()
    echo = which("echo")

    ever_config_new = EverestConfig.with_defaults(
        model={"realizations": [0]},
        install_jobs=[{"name": existing_job, "executable": echo}],
    )
    with ErtPluginContext() as runtime_plugins:
        runmodel_new = EverestRunModel.create(
            ever_config_new, runtime_plugins=runtime_plugins
        )

    only_fm_step = runmodel_new.forward_model_steps[0]
    assert only_fm_step.executable == echo
    assert only_fm_step.name == existing_job


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

    obj = ErtRuntimePlugins(
        queue_options=LsfQueueOptions(lsf_resource="my_resource", lsf_queue="my_queue")
    )
    with mock.patch(
        "ert.plugins.plugin_manager.ErtRuntimePlugins",
        return_value=obj,
    ):
        config = EverestConfig.load_file("config.yml")
        assert config.simulator.queue_system == LsfQueueOptions(
            lsf_queue="my_queue", lsf_resource="my_resource"
        )

        with ErtPluginContext() as runtime_plugins:
            queue_config = QueueConfig.from_dict(
                everest_to_ert_config_dict(config),
                site_queue_options=runtime_plugins.queue_options,
            )

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

    with ErtPluginContext() as runtime_plugins:
        runmodel = EverestRunModel.create(
            config, "some_exp_name", "batch", runtime_plugins=runtime_plugins
        )
    smry_config = next(
        r for r in runmodel.response_configuration if isinstance(r, SummaryConfig)
    )

    assert set(custom_sum_keys).issubset(set(smry_config.keys))


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
    "max_memory, exception_message",
    [
        ("1x", "Unknown memory unit"),
        ("1 x", "Unknown memory unit"),
        ("1 xy", "Unknown memory unit"),
        ("foo", "Invalid memory string: foo"),
    ],
)
def test_that_invalid_max_memory_fails(max_memory, exception_message) -> None:
    with pytest.raises(ValidationError, match=exception_message):
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


@pytest.mark.usefixtures("no_plugins")
@pytest.mark.parametrize(
    "max_memory, realization_memory",
    [(None, 0), (999, 999), ("1Gb", 1073741824), (0, 0)],
)
def test_that_max_memory_is_passed_to_realization_memory(
    change_to_tmpdir, max_memory, realization_memory
) -> None:
    config = EverestConfig.with_defaults(simulator={"max_memory": max_memory})
    assert config.simulator.queue_system.realization_memory == realization_memory


@pytest.mark.usefixtures("no_plugins")
@pytest.mark.parametrize(
    "max_memory, realization_memory, expected",
    [(None, 0, 0), (0, 0, 0), (111, 999, 999), (55, 0, 55)],
)
def test_that_max_memory_does_not_overwrite_realization_memory(
    change_to_tmpdir, max_memory, realization_memory, expected
) -> None:
    config = EverestConfig.with_defaults(
        simulator={
            "max_memory": max_memory,
            "queue_system": {"name": "local", "realization_memory": realization_memory},
        }
    )
    assert config.simulator.queue_system.realization_memory == expected


@pytest.mark.parametrize(
    "realization_memory, expected",
    [
        ("1Gb", 1073741824),
        ("2Kb", 2048),
        (999, 999),
    ],
)
def test_parsing_of_relization_memory(realization_memory, expected) -> None:
    config = EverestConfig.with_defaults(
        simulator={
            "queue_system": {"name": "local", "realization_memory": realization_memory},
        }
    )
    assert config.simulator.queue_system.realization_memory == expected


@pytest.mark.parametrize(
    "invalid_memory_spec, error_message",
    [
        ("-1", "Negative memory does not make sense"),
        ("      -2", "Negative memory does not make sense"),
        ("-1b", "Negative memory does not make sense in -1b"),
        ("b", "Invalid memory string"),
        ("'kljh3 k34f15gg.  asd '", "Invalid memory string"),
        ("'kljh3 1gb'", "Invalid memory string"),
        ("' 2gb 3k 1gb'", "Invalid memory string"),
        ("4ub", "Unknown memory unit"),
    ],
)
def test_parsing_of_invalid_relization_memory(
    invalid_memory_spec, error_message
) -> None:
    with pytest.raises(ValidationError, match=error_message):
        EverestConfig.with_defaults(
            simulator={
                "queue_system": {
                    "name": "local",
                    "realization_memory": invalid_memory_spec,
                },
            }
        )


def test_parsing_of_non_existing_relization_memory() -> None:
    config = EverestConfig.with_defaults(
        simulator={
            "queue_system": {"name": "local"},
        }
    )
    assert config.simulator.queue_system.realization_memory == 0


def test_that_everest_to_ert_config_dict_does_not_create_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    everest_to_ert_config_dict(EverestConfig.with_defaults())
    assert not os.listdir(tmp_path)


def test_that_export_keywords_are_turned_into_summary_config_keys(
    monkeypatch, tmp_path, min_config
):
    monkeypatch.chdir(tmp_path)
    extra_sum_keys = [
        "GOIR:PRODUC",
        "GOIT:INJECT",
        "GOIT:PRODUC",
        "GWPR:INJECT",
        "GWPR:PRODUC",
        "GWPT:INJECT",
        "GWPT:PRODUC",
        "GWIR:INJECT",
    ]

    min_config["export"] = {"keywords": extra_sum_keys}
    min_config["forward_model"] = [
        {
            "job": "eclipse100 CASE.DATA",
            "results": {"file_name": "CASE", "type": "summary"},
        }
    ]
    with ErtPluginContext() as runtime_plugins:
        config = EverestConfig(**min_config)
        runmodel = EverestRunModel.create(
            config, "exp", "batch", runtime_plugins=runtime_plugins
        )
    summary_config = next(
        r for r in runmodel.response_configuration if isinstance(r, SummaryConfig)
    )
    assert set(extra_sum_keys).issubset(summary_config.keys)


def test_that_summary_keys_are_passed_through_forward_model_results(
    monkeypatch, tmp_path, min_config
):
    monkeypatch.chdir(tmp_path)
    min_config["forward_model"] = [
        {
            "job": "eclipse100 CASE.DATA",
            "results": {
                "file_name": "CASE",
                "type": "summary",
                "keys": ["one", "two", "three"],
            },
        }
    ]

    with ErtPluginContext() as runtime_plugins:
        config = EverestConfig(**min_config)
        runmodel = EverestRunModel.create(
            config, "exp", "batch", runtime_plugins=runtime_plugins
        )

    summary_config = next(
        r for r in runmodel.response_configuration if isinstance(r, SummaryConfig)
    )

    assert {"one", "two", "three"}.issubset(summary_config.keys)


def test_that_summary_keys_default_to_expected_keys_according_to_wells(
    monkeypatch, tmp_path, min_config
):
    monkeypatch.chdir(tmp_path)
    min_config["forward_model"] = [
        {
            "job": "eclipse100 CASE.DATA",
            "results": {
                "file_name": "CASE",
                "type": "summary",
            },
        }
    ]
    min_config["wells"] = [{"name": "OP1"}, {"name": "WI1"}]
    min_config["controls"] = [
        {
            "name": "well_rate",
            "type": "well_control",
            "perturbation_magnitude": 0.01,
            "variables": [
                {
                    "name": "OP1",
                    "index": 1,
                    "initial_guess": 50,
                    "min": 10,
                    "max": 500,
                },
                {
                    "name": "WI1",
                    "index": 1,
                    "initial_guess": 250,
                    "min": 10,
                    "max": 500,
                },
            ],
        }
    ]

    with ErtPluginContext() as runtime_plugins:
        config = EverestConfig(**min_config)
        runmodel = EverestRunModel.create(
            config, "exp", "batch", runtime_plugins=runtime_plugins
        )

    summary_config = next(
        r for r in runmodel.response_configuration if isinstance(r, SummaryConfig)
    )

    data_keys = everest.simulator.DEFAULT_DATA_SUMMARY_KEYS
    field_keys = everest.simulator.DEFAULT_FIELD_SUMMARY_KEYS
    well_sum_keys = everest.simulator.DEFAULT_WELL_SUMMARY_KEYS

    expected_defaulted_sum_keys = (
        ["*"]
        + data_keys
        + field_keys
        + [":".join(tup) for tup in itertools.product(well_sum_keys, ["OP1", "WI1"])]
    )

    assert set(summary_config.keys) == set(expected_defaulted_sum_keys)


def test_that_install_data_raises_error_on_missing_copy_file(tmp_path):
    source_file = tmp_path / "some_file.json"
    source_file.write_text('{"mock_key": "mock_value"}')

    config = InstallDataConfig(source=str(source_file), target="the_output.json")

    with pytest.raises(KeyError, match=r"ERT forward model: copy_file to be installed"):
        config.to_ert_forward_model_step(
            config_directory=str(tmp_path),
            output_directory=str(tmp_path / "output"),
            model_realizations=[0],
            installed_fm_steps={},
        )


def test_that_install_data_raises_error_on_missing_copy_directory(tmp_path):
    config_directory = tmp_path / "config_dir"
    source_directory = config_directory / "<GEO_ID>"
    realizations = [0, 1, 2]

    for realization in realizations:
        realization_dir = source_directory.with_name(
            source_directory.name.replace("<GEO_ID>", str(realization))
        )
        realization_dir.mkdir(parents=True)

    config = InstallDataConfig(
        source=str(source_directory), target="target_dir", link=False
    )

    with pytest.raises(
        KeyError, match=r"ERT forward model: copy_directory to be installed"
    ):
        config.to_ert_forward_model_step(
            config_directory=str(config_directory),
            output_directory=str(tmp_path / "output"),
            model_realizations=realizations,
            installed_fm_steps={},
        )


def test_that_install_data_raises_error_on_missing_symlink(tmp_path):
    source_file = tmp_path / "source_file.json"
    source_file.write_text('{"mock_key": "mock_value"}')

    config = InstallDataConfig(
        source=str(source_file),
        target="linked_file.json",
        link=True,
    )

    with pytest.raises(KeyError, match=r"ERT forward model: symlink to be installed"):
        config.to_ert_forward_model_step(
            config_directory=str(tmp_path),
            output_directory=str(tmp_path / "output"),
            model_realizations=[0],
            installed_fm_steps={},
        )
