import itertools
import os
from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from ruamel.yaml import YAML

import everest
from ert.config.parsing import ConfigKeys as ErtConfigKeys
from everest import ConfigKeys as CK
from everest.config import EverestConfig
from everest.config.install_job_config import InstallJobConfig
from everest.config.well_config import WellConfig
from everest.config.workflow_config import WorkflowConfig
from everest.simulator.everest_to_ert import (
    _everest_to_ert_config_dict,
    everest_to_ert_config,
)
from tests.everest.utils import (
    hide_opm,
    skipif_no_everest_models,
    skipif_no_opm,
)

SNAKE_CONFIG_DIR = "snake_oil/everest/model"
SNAKE_CONFIG_PATH = os.path.join(SNAKE_CONFIG_DIR, "snake_oil.yml")
TUTORIAL_CONFIG_DIR = "mocked_test_case"


@pytest.mark.parametrize(
    "config, expected",
    [
        [
            {
                "queue_system": "torque",
                "name": "permanent_8",
                "qsub_cmd": "qsub",
                "qstat_cmd": "qstat",
                "qdel_cmd": "qdel",
                "keep_qsub_output": 1,
                "submit_sleep": 0.5,
                "project_code": "snake_oil_pc",
                "cores_per_node": 3,
            },
            {
                "project_code": "snake_oil_pc",
                "qsub_cmd": "qsub",
                "qstat_cmd": "qstat",
                "qdel_cmd": "qdel",
                "num_cpus_per_node": 3,
                "num_nodes": 1,
                "keep_qsub_output": True,
                "queue_name": "permanent_8",
            },
        ],
        [
            {
                "queue_system": "slurm",
                "name": "default-queue",
                "max_memory": "1000M",
                "exclude_host": "host1,host2,host3,host4",
                "include_host": "host5,host6,host7,host8",
            },
            {
                "exclude_hosts": "host1,host2,host3,host4",
                "include_hosts": "host5,host6,host7,host8",
                "memory": "1000M",
                "queue_name": "default-queue",
                "sbatch_cmd": "sbatch",
                "scancel_cmd": "scancel",
                "scontrol_cmd": "scontrol",
                "squeue_cmd": "squeue",
                "squeue_timeout": 2,
            },
        ],
        [
            {
                "queue_system": "lsf",
                "name": "mr",
                "options": "span = 1 && select[x86 and GNU/Linux]",
                "server": "lx-fastserver01",
            },
            {
                "queue_name": "mr",
                "resource_requirement": "span = 1 && select[x86 and GNU/Linux]",
            },
        ],
    ],
)
def test_everest_to_ert_queue_config(config, expected):
    general_options = {"resubmit_limit": 7, "cores": 42}
    ever_config = EverestConfig.with_defaults(
        **{
            "simulator": config | general_options,
            "model": {"realizations": [0]},
        }
    )
    ert_config = everest_to_ert_config(ever_config)

    qc = ert_config.queue_config
    qo = qc.queue_options
    assert qc.queue_system == config["queue_system"].upper()
    driver_options = qo.driver_options
    driver_options.pop("activate_script")
    assert {k: v for k, v in driver_options.items() if v is not None} == expected
    assert qc.max_submit == general_options["resubmit_limit"] + 1
    assert qo.max_running == general_options["cores"]


@pytest.mark.parametrize(
    "name",
    [
        "render",
        "recovery_factor",
        "wdreorder",
        "wdfilter",
        "wdupdate",
        "wdset",
        "wdcompl",
        "wddatefilter",
    ],
)
def test_default_installed_jobs(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    ever_config_dict = EverestConfig.with_defaults(
        **yaml.safe_load(
            dedent(f"""
    model: {{"realizations": [0]}}
    forward_model:
      - {name}
    """)
        )
    )
    config = everest_to_ert_config(ever_config_dict)
    # Index 0 is the copy job for wells.json
    assert config.forward_model_steps[1].name == name


@skipif_no_opm
def test_combined_wells_everest_to_ert(copy_test_data_to_tmp):
    config_mocked_multi_batch = os.path.join(
        TUTORIAL_CONFIG_DIR, "mocked_multi_batch.yml"
    )
    # Load config file
    ever_config_dict = EverestConfig.load_file(config_mocked_multi_batch)

    # Add a dummy well name to the everest config
    assert ever_config_dict.wells is not None
    ever_config_dict.wells.append(WellConfig(name="fakename"))
    ert_config_dict = _everest_to_ert_config_dict(ever_config_dict)

    # Check whether dummy name is in the summary keys
    fakename_in_strings = [
        "fakename" in string for string in ert_config_dict[ErtConfigKeys.SUMMARY][0]
    ]
    assert any(fakename_in_strings)

    # Check whether data file specific well is in the summary keys
    inj_in_strings = [
        "INJ" in string for string in ert_config_dict[ErtConfigKeys.SUMMARY][0]
    ]
    assert any(inj_in_strings)


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
        **{
            "model": {"realizations": [0]},
            "install_data": [{"source": source, "target": target, "link": symlink}],
        }
    )

    errors = EverestConfig.lint_config_dict(ever_config.to_dict())
    assert len(errors) == 0

    ert_config = everest_to_ert_config(ever_config)
    expected_fm = next(val for val in ert_config.forward_model_steps if val.name == cmd)
    assert expected_fm.arglist == (f"./{source}", target)


@skipif_no_opm
@skipif_no_everest_models
@pytest.mark.everest_models_test
@pytest.mark.integration_test
def test_summary_default(copy_egg_test_data_to_tmp):
    config_dir = "everest/model"
    config_file = os.path.join(config_dir, "config.yml")
    everconf = EverestConfig.load_file(config_file)

    data_file = everconf.model.data_file
    if not os.path.isabs(data_file):
        data_file = os.path.join(config_dir, data_file)
    data_file = data_file.replace("<GEO_ID>", "0")

    wells = everest.util.read_wellnames(data_file)
    groups = everest.util.read_groupnames(data_file)

    sum_keys = list(everest.simulator.DEFAULT_DATA_SUMMARY_KEYS) + list(
        everest.simulator.DEFAULT_FIELD_SUMMARY_KEYS
    )

    key_name_lists = (
        (everest.simulator.DEFAULT_GROUP_SUMMARY_KEYS, groups),
        (everest.simulator.DEFAULT_WELL_SUMMARY_KEYS, wells),
    )
    for keys, names in key_name_lists:
        sum_keys += [f"{key}:{name}" for key, name in itertools.product(keys, names)]

    res_conf = _everest_to_ert_config_dict(everconf)
    assert set(sum_keys) == set(res_conf[ErtConfigKeys.SUMMARY][0])


@pytest.mark.integration_test
@hide_opm
@skipif_no_everest_models
@pytest.mark.everest_models_test
@pytest.mark.fails_on_macos_github_workflow
def test_summary_default_no_opm(copy_egg_test_data_to_tmp):
    config_dir = "everest/model"
    config_file = os.path.join(config_dir, "config.yml")
    everconf = EverestConfig.load_file(config_file)

    # Read wells from the config instead of using opm
    wells = [w.name for w in everconf.wells]
    sum_keys = (
        list(everest.simulator.DEFAULT_DATA_SUMMARY_KEYS)
        + list(everest.simulator.DEFAULT_FIELD_SUMMARY_KEYS)
        + [
            "{}:{}".format(k, w)
            for k, w in itertools.product(
                everest.simulator.DEFAULT_WELL_SUMMARY_KEYS, wells
            )
        ]
    )
    sum_keys = [list(set(sum_keys))]
    res_conf = _everest_to_ert_config_dict(everconf)

    assert set(sum_keys[0]) == set(res_conf[ErtConfigKeys.SUMMARY][0])


@pytest.mark.parametrize(
    "install_data, expected_error_msg",
    [
        (
            {"source": "r{{ foo }}/", "link": True, "target": "bar.json"},
            "'/' is a mount point and can't be handled",
        ),
        (
            {"source": "baz/", "link": True, "target": "bar.json"},
            "No such file or directory",
        ),
        (
            {"source": None, "link": True, "target": "bar.json"},
            "Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]",
        ),
        (
            {"source": "", "link": "false", "target": "bar.json"},
            " false could not be parsed to a boolean",
        ),
        (
            {"source": "baz/", "link": True, "target": 3},
            "Input should be a valid string [type=string_type, input_value=3, input_type=int]",
        ),
    ],
)
def test_install_data_with_invalid_templates(
    copy_mocked_test_data_to_tmp,
    install_data,
    expected_error_msg,
):
    """
    Checks for InstallDataConfig's validations instantiating EverestConfig to also
    check invalid template rendering (e.g 'r{{ foo }}/) that maps to '/'
    """

    config_file = "mocked_multi_batch.yml"

    with open(config_file, encoding="utf-8") as f:
        raw_config = YAML(typ="safe", pure=True).load(f)

    raw_config[CK.INSTALL_DATA] = [install_data]

    with open(config_file, "w", encoding="utf-8") as f:
        yaml = YAML(typ="safe", pure=True)
        yaml.indent = 2
        yaml.default_flow_style = False
        yaml.dump(raw_config, f)

    with pytest.raises(ValueError) as exc_info:
        EverestConfig.load_file(config_file)

    assert expected_error_msg in str(exc_info.value)


def test_workflow_job(copy_snake_oil_to_tmp):
    workflow_jobs = [{"name": "test", "source": "jobs/TEST"}]
    ever_config = EverestConfig.load_file(SNAKE_CONFIG_PATH)
    ever_config.install_workflow_jobs = workflow_jobs
    ert_config_dict = _everest_to_ert_config_dict(ever_config)
    jobs = ert_config_dict.get(ErtConfigKeys.LOAD_WORKFLOW_JOB)
    assert jobs is not None
    assert jobs[0] == (
        os.path.join(ever_config.config_directory, workflow_jobs[0]["source"]),
        workflow_jobs[0]["name"],
    )


def test_workflows(copy_snake_oil_to_tmp):
    workflow_jobs = [{"name": "test", "source": "jobs/TEST"}]
    ever_config = EverestConfig.load_file(SNAKE_CONFIG_PATH)
    ever_config.install_workflow_jobs = workflow_jobs
    ever_config.workflows = WorkflowConfig.model_validate(
        {"pre_simulation": ["test -i in -o out"]}
    )
    ert_config_dict = _everest_to_ert_config_dict(ever_config)
    workflows = ert_config_dict.get(ErtConfigKeys.LOAD_WORKFLOW)
    assert workflows is not None
    name = os.path.join(ever_config.config_directory, ".pre_simulation.workflow")
    assert os.path.exists(name)
    assert workflows[0] == (name, "pre_simulation")
    hooks = ert_config_dict.get(ErtConfigKeys.HOOK_WORKFLOW)
    assert hooks is not None
    assert hooks[0] == ("pre_simulation", "PRE_SIMULATION")


def test_user_config_jobs_precedence(copy_snake_oil_to_tmp):
    # Load config file
    ever_config = EverestConfig.load_file(SNAKE_CONFIG_PATH)
    first_job = everest.jobs.script_names[0]

    existing_standard_job = InstallJobConfig(name=first_job, source="expected_source")
    ever_config.install_jobs.append(existing_standard_job)
    config_dir = ever_config.config_directory
    # Transform to res dict
    ert_config_dict = _everest_to_ert_config_dict(ever_config)

    job = [
        job for job in ert_config_dict[ErtConfigKeys.INSTALL_JOB] if job[0] == first_job
    ]
    assert len(job) == 1
    assert job[0][1] == os.path.join(config_dir, "expected_source")
