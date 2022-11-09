import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

from ert._c_wrappers.enkf import ResConfig


@pytest.fixture(name="minimum_config_dict")
@pytest.mark.usefixtures()
def fixture_minimum_config_dict():
    return {
        "INTERNALS": {
            "CONFIG_DIRECTORY": ".",
        },
        "SIMULATION": {
            "QUEUE_SYSTEM": {
                "JOBNAME": "Job%d",
            },
            "RUNPATH": "/tmp/simulations/run%d",
            "NUM_REALIZATIONS": 10,
            "MIN_REALIZATIONS": 10,
            "JOB_SCRIPT": "script.sh",
            "ENSPATH": "Ensemble",
        },
    }


def test_minimum_config(minimum_config_dict, minimum_case):
    loaded_res_config = minimum_case.resConfig()
    prog_res_config = ResConfig(config=minimum_config_dict)

    assert (
        loaded_res_config.model_config.num_realizations
        == prog_res_config.model_config.num_realizations
    )

    assert (
        loaded_res_config.model_config.jobname_format_string
        == prog_res_config.model_config.jobname_format_string
    )

    assert (
        loaded_res_config.model_config.runpath_format_string
        == prog_res_config.model_config.runpath_format_string
    )

    assert (
        loaded_res_config.model_config.ens_path == prog_res_config.model_config.ens_path
    )

    assert len(prog_res_config.errors) == 0
    assert len(prog_res_config.failed_keys) == 0


@pytest.mark.usefixtures("copy_minimum_case")
def test_errors():
    with pytest.raises(ValueError):
        ResConfig(
            config={
                "INTERNALS": {
                    "CONFIG_DIRECTORY": ".",
                },
                "SIMULATION": {
                    "QUEUE_SYSTEM": {
                        "JOBNAME": "Job%d",
                    },
                    "RUNPATH": "/tmp/simulations/run%d",
                    "NUM_REALIZATIONS": "/should/be/an/integer",
                    "JOB_SCRIPT": "script.sh",
                    "ENSPATH": "Ensemble",
                },
            }
        )


@pytest.mark.usefixtures("copy_minimum_case")
def test_failed_keys():
    res_config = ResConfig(
        config={
            "INTERNALS": {
                "CONFIG_DIRECTORY": ".",
            },
            "UNKNOWN_KEY": "Have/not/got/a/clue",
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "JOBNAME": "Job%d",
                },
                "RUNPATH": "/tmp/simulations/run%d",
                "NUM_REALIZATIONS": 10,
                "JOB_SCRIPT": "script.sh",
                "ENSPATH": "Ensemble",
            },
        }
    )

    assert len(res_config.failed_keys) == 1
    assert list(res_config.failed_keys.keys()) == ["UNKNOWN_KEY"]
    assert res_config.failed_keys["UNKNOWN_KEY"] == "Have/not/got/a/clue"


def test_new_config(monkeypatch):
    monkeypatch.chdir(tempfile.mkdtemp())
    with open("NEW_TYPE_A", mode="w", encoding="utf-8") as fout:
        fout.write(
            dedent(
                """
        EXECUTABLE echo
        MIN_ARG    1
        MAX_ARG    4
        ARG_TYPE 0 STRING
        ARG_TYPE 1 BOOL
        ARG_TYPE 2 FLOAT
        ARG_TYPE 3 INT
        """
            )
        )
    prog_res_config = ResConfig(
        config={
            "INTERNALS": {},
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "QUEUE_SYSTEM": "LOCAL",
                    "JOBNAME": "SIM_KW",
                },
                "ECLBASE": "SIM_KW",
                "NUM_REALIZATIONS": 10,
                "INSTALL_JOB": [
                    {"NAME": "NEW_JOB_A", "PATH": "NEW_TYPE_A"},
                ],
                "SIMULATION_JOB": [
                    {"NAME": "NEW_JOB_A", "ARGLIST": ["Hello", True, 3.14, 4]},
                    {"NAME": "NEW_JOB_A", "ARGLIST": ["word"]},
                ],
            },
        }
    )
    forward_model = prog_res_config.forward_model
    assert forward_model.jobs[0].get_arglist() == ["Hello", "True", "3.14", "4"]
    assert forward_model.jobs[1].get_arglist() == ["word"]


def test_large_config(setup_case):
    loaded_res_config = setup_case("snake_oil_structure", "ert/model/user_config.ert")
    prog_res_config = ResConfig(
        config={
            "DEFINES": {
                "<USER>": "TEST_USER",
                "<SCRATCH>": "ert/model/scratch/ert",
                "<CASE_DIR>": "the_extensive_case",
                "<ECLIPSE_NAME>": "XYZ",
            },
            "INTERNALS": {
                "CONFIG_DIRECTORY": ".",
            },
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "JOBNAME": "SNAKE_OIL_STRUCTURE_%d",
                    "QUEUE_SYSTEM": "LSF",
                    "MAX_RUNTIME": 23400,
                    "MIN_REALIZATIONS": "50%",
                    "MAX_SUBMIT": 13,
                    "QUEUE_OPTION": [
                        {
                            "DRIVER_NAME": "LSF",
                            "OPTION": "MAX_RUNNING",
                            "VALUE": 100,
                        },
                        {
                            "DRIVER_NAME": "LSF",
                            "OPTION": "LSF_RESOURCE",
                            "VALUE": "select[x86_64Linux] same[type:model]",
                        },
                        {
                            "DRIVER_NAME": "LSF",
                            "OPTION": "LSF_SERVER",
                            "VALUE": "simulacrum",
                        },
                        {
                            "DRIVER_NAME": "LSF",
                            "OPTION": "LSF_QUEUE",
                            "VALUE": "mr",
                        },
                    ],
                },
                "DATA_FILE": "eclipse/model/SNAKE_OIL.DATA",
                "RUNPATH": "<SCRATCH>/<USER>/<CASE_DIR>/realization-%d/iter-%d",
                "RUNPATH_FILE": "../output/run_path_file/.ert-runpath-list_<CASE_DIR>",
                "ECLBASE": "eclipse/model/<ECLIPSE_NAME>-%d",
                "NUM_REALIZATIONS": "10",
                "ENSPATH": "ert/output/storage/<CASE_DIR>",
                "GRID": "eclipse/include/grid/CASE.EGRID",
                "REFCASE": "ert/input/refcase/SNAKE_OIL_FIELD",
                "HISTORY_SOURCE": "REFCASE_HISTORY",
                "OBS_CONFIG": "ert/input/observations/obsfiles/observations.txt",
                "TIME_MAP": "ert/input/refcase/time_map.txt",
                "SUMMARY": [
                    "WOPR:PROD",
                    "WOPT:PROD",
                    "WWPR:PROD",
                    "WWCT:PROD",
                    "WWPT:PROD",
                    "WBHP:PROD",
                    "WWIR:INJ",
                    "WWIT:INJ",
                    "WBHP:INJ",
                    "ROE:1",
                ],
                "INSTALL_JOB": [
                    {
                        "NAME": "SNAKE_OIL_SIMULATOR",
                        "PATH": "snake_oil/jobs/SNAKE_OIL_SIMULATOR",
                    },
                    {
                        "NAME": "SNAKE_OIL_NPV",
                        "PATH": "snake_oil/jobs/SNAKE_OIL_NPV",
                    },
                    {
                        "NAME": "SNAKE_OIL_DIFF",
                        "PATH": "snake_oil/jobs/SNAKE_OIL_DIFF",
                    },
                ],
                "LOAD_WORKFLOW_JOB": ["ert/bin/workflows/workflowjobs/UBER_PRINT"],
                "LOAD_WORKFLOW": ["ert/bin/workflows/MAGIC_PRINT"],
                "FORWARD_MODEL": [
                    "SNAKE_OIL_SIMULATOR",
                    "SNAKE_OIL_NPV",
                    "SNAKE_OIL_DIFF",
                ],
                "RUN_TEMPLATE": [
                    "ert/input/templates/seed_template.txt",
                    "seed.txt",
                ],
                "GEN_KW": [
                    {
                        "NAME": "SIGMA",
                        "TEMPLATE": "ert/input/templates/sigma.tmpl",
                        "OUT_FILE": "coarse.sigma",
                        "PARAMETER_FILE": "ert/input/distributions/sigma.dist",
                    }
                ],
                "GEN_DATA": [
                    {
                        "NAME": "super_data",
                        "RESULT_FILE": "super_data_%d",
                        "REPORT_STEPS": 1,
                    }
                ],
                "LOGGING": {
                    "UPDATE_LOG_PATH": "../output/update_log/<CASE_DIR>",
                },
            },
        }
    )

    loaded_model_config = loaded_res_config.model_config
    prog_model_config = prog_res_config.model_config
    assert loaded_model_config.num_realizations == prog_model_config.num_realizations

    assert (
        loaded_model_config.jobname_format_string
        == prog_model_config.jobname_format_string
    )
    assert (
        loaded_model_config.runpath_format_string
        == prog_model_config.runpath_format_string
    )
    assert prog_model_config.ens_path == loaded_model_config.ens_path
    assert loaded_model_config.history_source == prog_model_config.history_source
    assert loaded_model_config.obs_config_file == prog_model_config.obs_config_file
    assert loaded_res_config.installed_jobs == prog_res_config.installed_jobs
    assert loaded_res_config.env_vars == prog_res_config.env_vars
    assert (
        loaded_res_config.forward_model.job_name_list()
        == prog_res_config.forward_model.job_name_list()
    )

    assert (
        loaded_res_config.ensemble_config._grid_file
        == prog_res_config.ensemble_config._grid_file
    )
    assert (
        loaded_res_config.ensemble_config._refcase_file
        == prog_res_config.ensemble_config._refcase_file
    )

    assert (
        Path(prog_res_config.analysis_config.get_log_path()).resolve()
        == Path(loaded_res_config.analysis_config.get_log_path()).resolve()
    )
    assert (
        prog_res_config.analysis_config.get_max_runtime()
        == loaded_res_config.analysis_config.get_max_runtime()
    )

    assert set(loaded_res_config.ensemble_config.alloc_keylist()) == set(
        prog_res_config.ensemble_config.alloc_keylist()
    )

    assert prog_res_config.ert_templates == loaded_res_config.ert_templates

    assert loaded_res_config.ert_workflow_list == prog_res_config.ert_workflow_list

    assert prog_res_config.queue_config == loaded_res_config.queue_config

    assert len(prog_res_config.failed_keys) == 0
