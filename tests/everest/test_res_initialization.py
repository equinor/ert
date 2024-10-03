import itertools
import os
from unittest.mock import patch

import pytest

import everest
from ert.config import ErtConfig
from everest import ConfigKeys
from everest.config import EverestConfig
from everest.config.install_data_config import InstallDataConfig
from everest.config.install_job_config import InstallJobConfig
from everest.config.well_config import WellConfig
from everest.config.workflow_config import WorkflowConfig
from everest.simulator.everest_to_ert import (
    _everest_to_ert_config_dict,
    everest_to_ert_config,
)
from everest.util.forward_models import collect_forward_models
from tests.everest.utils import (
    everest_default_jobs,
    hide_opm,
    relpath,
    skipif_no_everest_models,
    skipif_no_opm,
)

NO_PROJECT_RES = (
    os.environ.get("NO_PROJECT_RES", False),
    "Skipping tests when no access to /project/res",
)
SNAKE_CONFIG_DIR = "snake_oil/everest/model"
SNAKE_CONFIG_PATH = os.path.join(SNAKE_CONFIG_DIR, "snake_oil.yml")
TUTORIAL_CONFIG_DIR = "mocked_test_case"


def build_snake_dict(output_dir, queue_system, report_steps=False):
    # This is a tested config from ert corresponding to the
    # snake_oil

    def simulation_jobs():
        sim_jobs = [
            (
                "copy_file",
                os.path.realpath(
                    "snake_oil/everest/model/everest_output/"
                    ".internal_data/wells.json"
                ),
                "wells.json",
            ),
            ("snake_oil_simulator",),
            ("snake_oil_npv",),
            ("snake_oil_diff",),
        ]
        if report_steps:
            sim_jobs.insert(2, ("eclipse100",))
            sim_jobs.insert(
                3,
                (
                    "strip_dates",
                    "--summary",
                    "<ECLBASE>.UNSMRY",
                    "--dates",
                    "2000-1-1",
                    "2001-1-2",
                    "2002-1-1",
                ),
            )
        return sim_jobs

    def install_jobs():
        jobs = [
            (
                "snake_oil_diff",
                os.path.join(
                    os.path.abspath(SNAKE_CONFIG_DIR), "../../jobs/SNAKE_OIL_DIFF"
                ),
            ),
            (
                "snake_oil_simulator",
                os.path.join(
                    os.path.abspath(SNAKE_CONFIG_DIR), "../../jobs/SNAKE_OIL_SIMULATOR"
                ),
            ),
            (
                "snake_oil_npv",
                os.path.join(
                    os.path.abspath(SNAKE_CONFIG_DIR), "../../jobs/SNAKE_OIL_NPV"
                ),
            ),
            *everest_default_jobs(output_dir),
        ]

        return jobs

    def local_queue_system():
        return {
            "QUEUE_SYSTEM": "LOCAL",
            "QUEUE_OPTION": [
                ("LOCAL", "MAX_RUNNING", 8),
            ],
        }

    # For the test comparison to succeed the elements in the QUEUE_OPTION list must
    # come in the same order as the options in the _extract_slurm_options() function
    # in everest_to_ert_config.
    def slurm_queue_system():
        return {
            "QUEUE_SYSTEM": "SLURM",
            "QUEUE_OPTION": [
                (
                    "SLURM",
                    "PARTITION",
                    "default-queue",
                ),
                (
                    "SLURM",
                    "MEMORY",
                    "1000M",
                ),
                (
                    "SLURM",
                    "EXCLUDE_HOST",
                    "host1,host2,host3,host4",
                ),
                (
                    "SLURM",
                    "INCLUDE_HOST",
                    "host5,host6,host7,host8",
                ),
            ],
        }

    def make_queue_system(queue_system):
        if queue_system == ConfigKeys.LOCAL:
            return local_queue_system()
        elif queue_system == ConfigKeys.SLURM:
            return slurm_queue_system()

    def make_gen_data():
        return [
            (
                "snake_oil_nvp",
                "RESULT_FILE:snake_oil_nvp",
            ),
        ]

    ert_config = {
        "DEFINE": [("<CONFIG_PATH>", os.path.abspath(SNAKE_CONFIG_DIR))],
        "RUNPATH": os.path.join(
            output_dir,
            "simulations/<CASE_NAME>/geo_realization_<GEO_ID>/simulation_<IENS>",
        ),
        "RUNPATH_FILE": os.path.join(
            os.path.realpath("snake_oil/everest/model"),
            "everest_output/.res_runpath_list",
        ),
        "NUM_REALIZATIONS": 10000,
        "MAX_RUNTIME": 3600,
        "ECLBASE": "eclipse/ECL",
        "INSTALL_JOB": install_jobs(),
        "SIMULATION_JOB": simulation_jobs(),
        "ENSPATH": os.path.join(
            os.path.realpath("snake_oil/everest/model"),
            "everest_output/simulation_results",
        ),
        "GEN_DATA": make_gen_data(),
    }
    ert_config.update(make_queue_system(queue_system))
    return ert_config


def build_tutorial_dict(config_dir, output_dir):
    # Expected config extracted from unittest.mocked_test_case.yml
    return {
        "DEFINE": [("<CONFIG_PATH>", config_dir)],
        "NUM_REALIZATIONS": 10000,
        "MAX_RUNTIME": 3600,
        "ECLBASE": "eclipse/ECL",
        "RUNPATH": os.path.join(
            output_dir,
            "simulations_{}".format(os.environ.get("USER")),
            "<CASE_NAME>",
            "geo_realization_<GEO_ID>",
            "simulation_<IENS>",
        ),
        "RUNPATH_FILE": os.path.join(
            os.path.realpath("mocked_test_case"),
            "everest_output/.res_runpath_list",
        ),
        "RANDOM_SEED": 999,
        "INSTALL_JOB": [
            ("well_order", os.path.join(config_dir, "jobs/WELL_ORDER_MOCK")),
            ("res_mock", os.path.join(config_dir, "jobs/RES_MOCK")),
            ("npv_function", os.path.join(config_dir, "jobs/NPV_FUNCTION_MOCK")),
            *everest_default_jobs(output_dir),
        ],
        "SIMULATION_JOB": [
            (
                "copy_file",
                os.path.realpath(
                    "mocked_test_case/everest_output/" ".internal_data/wells.json"
                ),
                "wells.json",
            ),
            (
                "well_order",
                "well_order.json",
                "SCHEDULE.INC",
                "ordered_wells.json",
            ),
            ("res_mock", "MOCKED_TEST_CASE"),
            ("npv_function", "MOCKED_TEST_CASE", "npv_function"),
        ],
        # Defaulted
        "QUEUE_SYSTEM": "LOCAL",
        "QUEUE_OPTION": [("LOCAL", "MAX_RUNNING", 8)],
        "ENSPATH": os.path.join(
            os.path.realpath("mocked_test_case"),
            "everest_output/simulation_results",
        ),
        "GEN_DATA": [
            (
                "npv_function",
                "RESULT_FILE:npv_function",
            ),
        ],
    }


def test_snake_everest_to_ert(copy_test_data_to_tmp):
    # Load config file
    ever_config_dict = EverestConfig.load_file(SNAKE_CONFIG_PATH)

    output_dir = ever_config_dict.output_dir
    snake_dict = build_snake_dict(output_dir, ConfigKeys.LOCAL)

    # Transform to res dict and verify equality
    ert_config_dict = _everest_to_ert_config_dict(ever_config_dict)
    assert snake_dict == ert_config_dict

    # Instantiate res
    ErtConfig.with_plugins().from_dict(
        config_dict=_everest_to_ert_config_dict(
            ever_config_dict, site_config=ErtConfig.read_site_config()
        )
    )


def test_snake_everest_to_ert_slurm(copy_test_data_to_tmp):
    snake_slurm_config_path = os.path.join(SNAKE_CONFIG_DIR, "snake_oil_slurm.yml")
    # Load config file
    ever_config_dict = EverestConfig.load_file(snake_slurm_config_path)

    output_dir = ever_config_dict.output_dir
    snake_dict = build_snake_dict(output_dir, ConfigKeys.SLURM)

    # Transform to res dict and verify equality
    ert_config_dict = _everest_to_ert_config_dict(ever_config_dict)
    assert snake_dict == ert_config_dict

    # Instantiate res
    ErtConfig.with_plugins().from_dict(
        config_dict=_everest_to_ert_config_dict(
            ever_config_dict, site_config=ErtConfig.read_site_config()
        )
    )


def test_snake_everest_to_ert_torque(copy_test_data_to_tmp):
    snake_torque_config_path = os.path.join(SNAKE_CONFIG_DIR, "snake_oil_torque.yml")

    ever_config = EverestConfig.load_file(snake_torque_config_path)
    ert_config_dict = _everest_to_ert_config_dict(ever_config)

    assert ert_config_dict["QUEUE_SYSTEM"] == "TORQUE"

    expected_queue_option_tuples = {
        ("TORQUE", "QSUB_CMD", "qsub"),
        ("TORQUE", "QSTAT_CMD", "qstat"),
        ("TORQUE", "QDEL_CMD", "qdel"),
        ("TORQUE", "QUEUE", "permanent_8"),
        ("TORQUE", "MEMORY_PER_JOB", "100mb"),
        ("TORQUE", "KEEP_QSUB_OUTPUT", 1),
        ("TORQUE", "SUBMIT_SLEEP", 0.5),
        ("TORQUE", "PROJECT_CODE", "snake_oil_pc"),
    }

    assert set(ert_config_dict["QUEUE_OPTION"]) == expected_queue_option_tuples

    ert_config = everest_to_ert_config(ever_config)

    qc = ert_config.queue_config
    qo = qc.queue_options
    assert qc.queue_system == "TORQUE"
    assert {k: v for k, v in qo.driver_options.items() if v is not None} == {
        "project_code": "snake_oil_pc",
        "qsub_cmd": "qsub",
        "qstat_cmd": "qstat",
        "qdel_cmd": "qdel",
        "memory_per_job": "100mb",
        "num_cpus_per_node": 1,
        "num_nodes": 1,
        "keep_qsub_output": True,
        "queue_name": "permanent_8",
    }


@patch.dict("os.environ", {"USER": "NO_USERNAME"})
def test_tutorial_everest_to_ert(copy_test_data_to_tmp):
    tutorial_config_path = os.path.join(TUTORIAL_CONFIG_DIR, "mocked_test_case.yml")
    # Load config file
    ever_config_dict = EverestConfig.load_file(tutorial_config_path)

    output_dir = ever_config_dict.output_dir
    tutorial_dict = build_tutorial_dict(
        os.path.abspath(TUTORIAL_CONFIG_DIR), output_dir
    )

    # Transform to res dict and verify equality
    ert_config_dict = _everest_to_ert_config_dict(ever_config_dict)
    assert tutorial_dict == ert_config_dict

    # Instantiate res
    ErtConfig.with_plugins().from_dict(
        config_dict=_everest_to_ert_config_dict(
            ever_config_dict, site_config=ErtConfig.read_site_config()
        )
    )


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
        "fakename" in string for string in ert_config_dict["SUMMARY"][0]
    ]
    assert any(fakename_in_strings)

    # Check whether data file specific well is in the summary keys
    inj_in_strings = ["INJ" in string for string in ert_config_dict["SUMMARY"][0]]
    assert any(inj_in_strings)


def test_lsf_queue_system(copy_test_data_to_tmp):
    snake_all_path = os.path.join(SNAKE_CONFIG_DIR, "snake_oil_all.yml")
    ever_config = EverestConfig.load_file(snake_all_path)

    assert ever_config.simulator.queue_system == ConfigKeys.LSF

    ert_config = _everest_to_ert_config_dict(ever_config)

    queue_system = ert_config["QUEUE_SYSTEM"]
    assert queue_system == "LSF"


def test_queue_configuration(copy_test_data_to_tmp):
    snake_all_path = os.path.join(SNAKE_CONFIG_DIR, "snake_oil_all.yml")
    ever_config = EverestConfig.load_file(snake_all_path)

    assert ever_config.simulator.cores == 3

    ert_config = _everest_to_ert_config_dict(ever_config)

    assert ert_config["MAX_SUBMIT"] == 17 + 1

    expected_options = [
        ("LSF", "MAX_RUNNING", 3),
        ("LSF", "LSF_QUEUE", "mr"),
        ("LSF", "LSF_SERVER", "lx-fastserver01"),
        ("LSF", "LSF_RESOURCE", "span = 1 && select[x86 and GNU/Linux]"),
    ]

    options = ert_config["QUEUE_OPTION"]
    assert options == expected_options


def test_queue_config():
    config_file = relpath("test_data/snake_oil/", "everest/model/snake_oil_all.yml")

    config = EverestConfig.load_file(config_file)

    assert config.simulator.name == "mr"
    assert config.simulator.resubmit_limit == 17
    assert config.simulator.cores == 3
    assert config.simulator.queue_system == "lsf"
    assert config.simulator.server == "lx-fastserver01"
    opts = "span = 1 && select[x86 and GNU/Linux]"
    assert opts == config.simulator.options


@patch.dict("os.environ", {"USER": "NO_USERNAME"})
def test_install_data_no_init(copy_test_data_to_tmp):
    """
    TODO: When default jobs are handled in Everest this test should be
    deleted as it is superseded by test_install_data.
    """
    sources = 2 * ["eclipse/refcase/TNO_REEK.SMSPEC"] + 2 * ["eclipse/refcase"]
    targets = 2 * ["REEK.SMSPEC"] + 2 * ["tno_refcase"]
    links = [True, False, True, False]
    cmd_list = ["symlink", "copy_file", "symlink", "copy_directory"]
    test_base = list(zip(sources, targets, links, cmd_list))
    tutorial_config_path = os.path.join(TUTORIAL_CONFIG_DIR, "mocked_test_case.yml")
    for source, target, link, cmd in test_base[1:2]:
        ever_config = EverestConfig.load_file(tutorial_config_path)

        if ever_config.install_data is None:
            ever_config.install_data = []

        ever_config.install_data.append(
            InstallDataConfig(
                source=source,
                target=target,
                link=link,
            )
        )

        errors = EverestConfig.lint_config_dict(ever_config.to_dict())
        assert len(errors) == 0

        ert_config_dict = _everest_to_ert_config_dict(ever_config)

        output_dir = ever_config.output_dir
        tutorial_dict = build_tutorial_dict(
            os.path.abspath(TUTORIAL_CONFIG_DIR), output_dir
        )

        config_dir = ever_config.config_directory
        tutorial_dict["SIMULATION_JOB"].insert(
            0,
            (cmd, os.path.join(config_dir, source), target),
        )
        assert tutorial_dict == ert_config_dict


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
    assert set(sum_keys) == set(res_conf["SUMMARY"][0])


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

    assert set(sum_keys[0]) == set(res_conf["SUMMARY"][0])


@pytest.mark.requires_eclipse
def test_install_data(copy_test_data_to_tmp):
    """
    TODO: When default jobs are handled in Everest this test should not
    be a simulation test.
    """

    sources = 2 * ["eclipse/refcase/TNO_REEK.SMSPEC"] + 2 * ["eclipse/refcase"]
    targets = 2 * ["REEK.SMSPEC"] + 2 * ["tno_refcase"]
    links = [True, False, True, False]
    cmds = ["symlink", "copy_file", "symlink", "copy_directory"]
    test_base = zip(sources, targets, links, cmds)
    tutorial_config_path = os.path.join(TUTORIAL_CONFIG_DIR, "mocked_test_case.yml")
    for source, target, link, cmd in test_base:
        ever_config = EverestConfig.load_file(tutorial_config_path)

        if ever_config.install_data is None:
            ever_config.install_data = []

        ever_config.install_data.append(
            InstallDataConfig(
                source=source,
                target=target,
                link=link,
            )
        )

        errors = EverestConfig.lint_config_dict(ever_config.to_dict())
        assert len(errors) == 0

        ert_config_dict = _everest_to_ert_config_dict(ever_config)

        output_dir = ever_config.output_dir
        tutorial_dict = build_tutorial_dict(
            os.path.abspath(TUTORIAL_CONFIG_DIR), output_dir
        )
        config_dir = ever_config.config_directory
        tutorial_dict["SIMULATION_JOB"].insert(
            0,
            (cmd, os.path.join(config_dir, source), target),
        )
        assert tutorial_dict == ert_config_dict

        # Instantiate res
        ErtConfig.with_plugins().from_dict(
            config_dict=_everest_to_ert_config_dict(
                ever_config, site_config=ErtConfig.read_site_config()
            )
        )


def test_strip_date_job_insertion(copy_test_data_to_tmp):
    # Load config file
    ever_config = EverestConfig.load_file(SNAKE_CONFIG_PATH)
    ever_config.model.report_steps = [
        "2000-1-1",
        "2001-1-2",
        "2002-1-1",
    ]
    ever_config.forward_model.insert(1, "eclipse100")

    output_dir = ever_config.output_dir
    snake_dict = build_snake_dict(output_dir, ConfigKeys.LOCAL, report_steps=True)

    # Transform to res dict and verify equality
    ert_config_dict = _everest_to_ert_config_dict(ever_config)
    assert snake_dict == ert_config_dict


def test_forward_model_job_insertion(copy_test_data_to_tmp):
    # Load config file
    ever_config = EverestConfig.load_file(SNAKE_CONFIG_PATH)

    # Transform to res dict
    ert_config_dict = _everest_to_ert_config_dict(ever_config)

    jobs = ert_config_dict["INSTALL_JOB"]
    for job in collect_forward_models():
        res_job = (job["name"], job["path"])
        assert res_job in jobs


def test_workflow_job(copy_test_data_to_tmp):
    workflow_jobs = [{"name": "test", "source": "jobs/TEST"}]
    ever_config = EverestConfig.load_file(SNAKE_CONFIG_PATH)
    ever_config.install_workflow_jobs = workflow_jobs
    ert_config_dict = _everest_to_ert_config_dict(ever_config)
    jobs = ert_config_dict.get("LOAD_WORKFLOW_JOB")
    assert jobs is not None
    assert jobs[0] == (
        os.path.join(ever_config.config_directory, workflow_jobs[0]["source"]),
        workflow_jobs[0]["name"],
    )


def test_workflows(copy_test_data_to_tmp):
    workflow_jobs = [{"name": "test", "source": "jobs/TEST"}]
    ever_config = EverestConfig.load_file(SNAKE_CONFIG_PATH)
    ever_config.install_workflow_jobs = workflow_jobs
    ever_config.workflows = WorkflowConfig.model_validate(
        {"pre_simulation": ["test -i in -o out"]}
    )
    ert_config_dict = _everest_to_ert_config_dict(ever_config)
    workflows = ert_config_dict.get("LOAD_WORKFLOW")
    assert workflows is not None
    name = os.path.join(ever_config.config_directory, ".pre_simulation.workflow")
    assert os.path.exists(name)
    assert workflows[0] == (name, "pre_simulation")
    hooks = ert_config_dict.get("HOOK_WORKFLOW")
    assert hooks is not None
    assert hooks[0] == ("pre_simulation", "PRE_SIMULATION")


def test_user_config_jobs_precedence(copy_test_data_to_tmp):
    # Load config file
    ever_config = EverestConfig.load_file(SNAKE_CONFIG_PATH)
    first_job = everest.jobs.script_names[0]

    existing_standard_job = InstallJobConfig(name=first_job, source="expected_source")
    ever_config.install_jobs.append(existing_standard_job)
    config_dir = ever_config.config_directory
    # Transform to res dict
    ert_config_dict = _everest_to_ert_config_dict(ever_config)

    job = [job for job in ert_config_dict["INSTALL_JOB"] if job[0] == first_job]
    assert len(job) == 1
    assert job[0][1] == os.path.join(config_dir, "expected_source")


def test_user_config_num_cpu(copy_test_data_to_tmp):
    # Load config file
    ever_config = EverestConfig.load_file(SNAKE_CONFIG_PATH)

    # Transform to res dict
    ert_config_dict = _everest_to_ert_config_dict(ever_config)
    assert "NUM_CPU" not in ert_config_dict

    ever_config.simulator.cores_per_node = 2
    # Transform to res dict
    ert_config_dict = _everest_to_ert_config_dict(ever_config)
    assert "NUM_CPU" in ert_config_dict
    assert ert_config_dict["NUM_CPU"] == 2
