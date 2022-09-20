#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_res_config.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
import os
import os.path
import stat
from datetime import date
from pathlib import Path

import pytest
from cwrap import Prototype, load
from ecl.util.enums import RngAlgTypeEnum

from ert._c_wrappers.enkf import (
    AnalysisConfig,
    ConfigKeys,
    GenDataFileType,
    QueueConfig,
    ResConfig,
    SiteConfig,
)
from ert._c_wrappers.job_queue import QueueDriverEnum
from ert._c_wrappers.sched import HistorySourceEnum

# The res_config object should set the environment variable
# 'DATA_ROOT' to the root directory with the config
# file. Unfortunately the python methods to get environment variable,
# os.getenv() and os.environ[] do not reflect the:
#
#    setenv( "DATA_ROOT" , ...)
#
# call in the res_config C code. We therefore create a wrapper to the
# underlying libc getenv() function to be used for testing.


clib = load(None)
clib_getenv = Prototype(clib, "char* getenv( char* )", bind=False)


config_defines = {
    "<USER>": "TEST_USER",
    "<SCRATCH>": "ert/model/scratch/ert",
    "<CASE_DIR>": "the_extensive_case",
    "<ECLIPSE_NAME>": "XYZ",
}

snake_oil_structure_config = {
    "RUNPATH": "<SCRATCH>/<USER>/<CASE_DIR>/realization-%d/iter-%d",
    "NUM_REALIZATIONS": 10,
    "MAX_RUNTIME": 23400,
    "MIN_REALIZATIONS": "50%",
    "MAX_SUBMIT": 13,
    "QUEUE_SYSTEM": "LSF",
    "LSF_QUEUE": "mr",
    "LSF_SERVER": "simulacrum",
    "LSF_RESOURCE": "select[x86_64Linux] same[type:model]",
    "UMASK": int("007", 8),
    "MAX_RUNNING": "100",
    "DATA_FILE": "eclipse/model/SNAKE_OIL.DATA",
    "START": date(2017, 1, 1),
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
    "GEN_KW": ["SIGMA"],
    "GEN_DATA": ["super_data"],
    "ECLBASE": "eclipse/model/<ECLIPSE_NAME>-%d",
    "ENSPATH": "ert/output/storage/<CASE_DIR>",
    "PLOT_PATH": "ert/output/results/plot/<CASE_DIR>",
    "UPDATE_LOG_PATH": "../output/update_log/<CASE_DIR>",
    "RUNPATH_FILE": "ert/output/run_path_file/.ert-runpath-list_<CASE_DIR>",
    "REFCASE": "ert/input/refcase/SNAKE_OIL_FIELD",
    "SIGMA": {
        "TEMPLATE": "ert/input/templates/sigma.tmpl",
        "RESULT": "coarse.sigma",
        "PARAMETER": "ert/input/distributions/sigma.dist",
    },
    "JOBNAME": "SNAKE_OIL_STRUCTURE_%d",
    "INSTALL_JOB": {
        "SNAKE_OIL_SIMULATOR": {
            "CONFIG": "snake_oil/jobs/SNAKE_OIL_SIMULATOR",
            "STDOUT": "snake_oil.stdout",
            "STDERR": "snake_oil.stderr",
            "EXECUTABLE": "snake_oil_simulator.py",
        },
        "SNAKE_OIL_NPV": {
            "CONFIG": "snake_oil/jobs/SNAKE_OIL_NPV",
            "STDOUT": "snake_oil_npv.stdout",
            "STDERR": "snake_oil_npv.stderr",
            "EXECUTABLE": "snake_oil_npv.py",
        },
        "SNAKE_OIL_DIFF": {
            "CONFIG": "snake_oil/jobs/SNAKE_OIL_DIFF",
            "STDOUT": "snake_oil_diff.stdout",
            "STDERR": "snake_oil_diff.stderr",
            "EXECUTABLE": "snake_oil_diff.py",
        },
    },
    "FORWARD_MODEL": ["SNAKE_OIL_SIMULATOR", "SNAKE_OIL_NPV", "SNAKE_OIL_DIFF"],
    "HISTORY_SOURCE": HistorySourceEnum.REFCASE_HISTORY,
    "OBS_CONFIG": "ert/input/observations/obsfiles/observations.txt",
    "LOAD_WORKFLOW": {"MAGIC_PRINT": "ert/bin/workflows/MAGIC_PRINT"},
    "LOAD_WORKFLOW_JOB": {
        "UBER_PRINT": "ert/bin/workflows/workflowjobs/bin/uber_print.py"
    },
    "RNG_ALG_TYPE": RngAlgTypeEnum.MZRAN,
    "GRID": "eclipse/include/grid/CASE.EGRID",
    "RUN_TEMPLATE": {
        "seed_template": {
            "TEMPLATE_FILE": "ert/input/templates/seed_template.txt",
            "TARGET_FILE": "seed.txt",
        }
    },
}

# Expand all strings in snake_oil_structure_config according to config_defines.
for define_key, define_value in config_defines.items():
    for data_key, data_value in snake_oil_structure_config.items():
        if isinstance(data_value, str):
            snake_oil_structure_config[data_key] = data_value.replace(
                define_key, define_value
            )


def test_invalid_user_config():
    with pytest.raises(IOError):
        ResConfig("this/is/not/a/file")


def test_missing_config():
    with pytest.raises(
        ValueError,
        match="Error trying to create ResConfig without any configuration",
    ):
        ResConfig()


def test_multiple_configs():
    with pytest.raises(
        ValueError,
        match="Attempting to create ResConfig object with multiple config objects",
    ):
        ResConfig(user_config_file="test", config="test")


def test_missing_directory():
    with pytest.raises(IOError):
        ResConfig(
            config={
                "INTERNALS": {
                    "CONFIG_DIRECTORY": "does_not_exist",
                },
                "SIMULATION": {
                    "QUEUE_SYSTEM": {
                        "JOBNAME": "Job%d",
                    },
                    "RUNPATH": "/tmp/simulations/run%d",
                    "NUM_REALIZATIONS": 1,
                    "JOB_SCRIPT": "script.sh",
                    "ENSPATH": "Ensemble",
                },
            }
        )


def test_init(minimum_case):
    res_config = minimum_case.resConfig()
    assert res_config.model_config.data_root() == os.getcwd()
    assert clib_getenv("DATA_ROOT") == os.getcwd()

    # This fails with an not-understandable Python error:
    # -----------------------------------------------------------------
    # res_config.model_config.set_data_root( "NEW" )
    # assert  res_config.model_config.data_root( )  == "NEW"
    # assert  clib_getenv("DATA_ROOT")  == "NEW"

    assert res_config is not None

    assert res_config.site_config is not None
    assert isinstance(res_config.site_config, SiteConfig)

    assert res_config.analysis_config is not None
    assert isinstance(res_config.analysis_config, AnalysisConfig)

    assert res_config.config_path == os.getcwd()

    assert res_config.subst_config["<CONFIG_PATH>"] == os.getcwd()


def test_extensive_config(setup_case):
    res_config = setup_case("local/snake_oil_structure", "ert/model/user_config.ert")

    model_config = res_config.model_config
    assert (
        Path(snake_oil_structure_config["RUNPATH"]).resolve()
        == Path(model_config.getRunpathAsString()).resolve()
    )
    assert (
        Path(snake_oil_structure_config["ENSPATH"]).resolve()
        == Path(model_config.getEnspath()).resolve()
    )
    assert snake_oil_structure_config["JOBNAME"] == model_config.getJobnameFormat()
    assert (
        snake_oil_structure_config["FORWARD_MODEL"]
        == model_config.getForwardModel().joblist()
    )
    assert (
        snake_oil_structure_config["HISTORY_SOURCE"]
        == model_config.get_history_source()
    )
    assert (
        snake_oil_structure_config["NUM_REALIZATIONS"] == model_config.num_realizations
    )
    assert (
        Path(snake_oil_structure_config["OBS_CONFIG"]).resolve()
        == Path(model_config.obs_config_file).resolve()
    )

    analysis_config = res_config.analysis_config
    assert (
        snake_oil_structure_config["MAX_RUNTIME"] == analysis_config.get_max_runtime()
    )
    assert (
        Path(snake_oil_structure_config["UPDATE_LOG_PATH"]).resolve()
        == Path(analysis_config.get_log_path()).resolve()
    )

    queue_config = res_config.queue_config
    for key, act in [
        ("MAX_SUBMIT", queue_config.max_submit),
        ("LSF_QUEUE", queue_config.queue_name),
        ("LSF_SERVER", queue_config.lsf_server),
        ("LSF_RESOURCE", queue_config.lsf_resource),
        ("QUEUE_SYSTEM", queue_config.queue_system),
        ("QUEUE_SYSTEM", queue_config.driver.name),
        ("MAX_RUNNING", queue_config.driver.get_option("MAX_RUNNING")),
    ]:
        assert snake_oil_structure_config[key] == act

    site_config = res_config.site_config
    assert site_config.umask == snake_oil_structure_config["UMASK"]
    job_list = site_config.get_installed_jobs()
    for job_name in snake_oil_structure_config["INSTALL_JOB"]:
        assert job_name in job_list

        exp_job_data = snake_oil_structure_config["INSTALL_JOB"][job_name]

        assert (
            Path(exp_job_data["CONFIG"]).resolve()
            == Path(job_list[job_name].get_config_file()).resolve()
        )
        assert exp_job_data["STDERR"] == job_list[job_name].get_stderr_file()
        assert exp_job_data["STDOUT"] == job_list[job_name].get_stdout_file()

    ecl_config = res_config.ecl_config
    assert (
        Path(snake_oil_structure_config["DATA_FILE"]).resolve()
        == Path(ecl_config.getDataFile()).resolve()
    )
    for extension in ["SMSPEC", "UNSMRY"]:
        assert (
            Path(snake_oil_structure_config["REFCASE"] + "." + extension).resolve()
            == Path(ecl_config.getRefcaseName() + "." + extension).resolve()
        )
    assert (
        Path(snake_oil_structure_config["GRID"]).resolve()
        == Path(ecl_config.get_gridfile()).resolve()
    )

    ensemble_config = res_config.ensemble_config
    assert set(
        snake_oil_structure_config["SUMMARY"]
        + snake_oil_structure_config["GEN_KW"]
        + snake_oil_structure_config["GEN_DATA"]
    ) == set(ensemble_config.alloc_keylist())

    assert (
        Path(snake_oil_structure_config["SIGMA"]["TEMPLATE"]).resolve()
        == Path(
            ensemble_config["SIGMA"].getKeywordModelConfig().getTemplateFile()
        ).resolve()
    )
    assert (
        Path(snake_oil_structure_config["SIGMA"]["PARAMETER"]).resolve()
        == Path(
            ensemble_config["SIGMA"].getKeywordModelConfig().getParameterFile()
        ).resolve()
    )
    assert (
        Path(snake_oil_structure_config["SIGMA"]["RESULT"]).resolve()
        == Path(ensemble_config["SIGMA"]._get_enkf_outfile()).resolve()
    )

    ert_workflow_list = res_config.ert_workflow_list
    assert len(snake_oil_structure_config["LOAD_WORKFLOW"]) == len(
        ert_workflow_list.getWorkflowNames()
    )

    for w_name in snake_oil_structure_config["LOAD_WORKFLOW"]:
        assert w_name in ert_workflow_list
        assert (
            Path(snake_oil_structure_config["LOAD_WORKFLOW"][w_name]).resolve()
            == Path(ert_workflow_list[w_name].src_file).resolve()
        )

    for wj_name in snake_oil_structure_config["LOAD_WORKFLOW_JOB"]:
        assert ert_workflow_list.hasJob(wj_name)
        job = ert_workflow_list.getJob(wj_name)

        assert wj_name == job.name()
        assert (
            Path(snake_oil_structure_config["LOAD_WORKFLOW_JOB"][wj_name]).resolve()
            == Path(job.executable()).resolve()
        )

    rng_config = res_config.rng_config
    assert snake_oil_structure_config["RNG_ALG_TYPE"] == rng_config.alg_type


def test_res_config_dict_constructor(setup_case):
    _ = setup_case("local/snake_oil_structure", "ert/model/user_config.ert")
    # create script file
    script_file = "script.sh"
    with open(script_file, "w") as f:
        f.write("""#!/bin/sh\nls""")

    st = os.stat(script_file)
    os.chmod(script_file, stat.S_IEXEC | st.st_mode)

    # split config_file to path and filename
    cfg_path, cfg_file = os.path.split(os.path.realpath("ert/model/user_config.ert"))

    config_data_new = {
        ConfigKeys.ALPHA_KEY: 3,
        ConfigKeys.RERUN_KEY: False,
        ConfigKeys.RERUN_START_KEY: 0,
        ConfigKeys.STD_CUTOFF_KEY: 1e-6,
        ConfigKeys.STOP_LONG_RUNNING: False,
        ConfigKeys.SINGLE_NODE_UPDATE: False,
        ConfigKeys.GLOBAL_STD_SCALING: 1,
        ConfigKeys.MIN_REALIZATIONS: 5,
        # "MIN_REALIZATIONS"  : "50%", percentages need to be fixed or removed
        ConfigKeys.RUNPATH: "<SCRATCH>/<USER>/<CASE_DIR>/realization-%d/iter-%d",
        ConfigKeys.NUM_REALIZATIONS: 10,  # model
        ConfigKeys.MAX_RUNTIME: 23400,
        ConfigKeys.JOB_SCRIPT: "../../../script.sh",
        ConfigKeys.QUEUE_SYSTEM: QueueDriverEnum.LSF_DRIVER,
        ConfigKeys.USER_MODE: True,
        ConfigKeys.MAX_SUBMIT: 13,
        ConfigKeys.NUM_CPU: 0,
        ConfigKeys.QUEUE_OPTION: [
            {ConfigKeys.NAME: "MAX_RUNNING", ConfigKeys.VALUE: "100"},
            {ConfigKeys.NAME: QueueConfig.LSF_QUEUE_NAME_KEY, ConfigKeys.VALUE: "mr"},
            {
                ConfigKeys.NAME: QueueConfig.LSF_SERVER_KEY,
                ConfigKeys.VALUE: "simulacrum",
            },
            {
                ConfigKeys.NAME: QueueConfig.LSF_RESOURCE_KEY,
                ConfigKeys.VALUE: "select[x86_64Linux] same[type:model]",
            },
        ],
        ConfigKeys.UMASK: int("007", 8),
        ConfigKeys.MAX_RUNNING: "100",
        ConfigKeys.DATA_FILE: "../../eclipse/model/SNAKE_OIL.DATA",
        # "START"             : date(2017, 1, 1), no clue where this comes from
        ConfigKeys.GEN_KW_TAG_FORMAT: "<%s>",
        ConfigKeys.SUMMARY: [
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
        ],  # ensemble
        ConfigKeys.GEN_KW: [
            {
                ConfigKeys.NAME: "SIGMA",
                ConfigKeys.TEMPLATE: "../input/templates/sigma.tmpl",
                ConfigKeys.OUT_FILE: "coarse.sigma",
                ConfigKeys.PARAMETER_FILE: "../input/distributions/sigma.dist",
                ConfigKeys.INIT_FILES: None,
                ConfigKeys.MIN_STD: None,
                ConfigKeys.FORWARD_INIT: False,
            }  # ensemble
        ],
        ConfigKeys.GEN_DATA: [
            {
                ConfigKeys.NAME: "super_data",
                ConfigKeys.INPUT_FORMAT: GenDataFileType.ASCII,
                ConfigKeys.RESULT_FILE: "super_data_%d",
                ConfigKeys.REPORT_STEPS: [1],
                ConfigKeys.INIT_FILES: None,
                ConfigKeys.ECL_FILE: None,
                ConfigKeys.TEMPLATE: None,
                ConfigKeys.KEY_KEY: None,
            }  # ensemble
        ],
        ConfigKeys.ECLBASE: "eclipse/model/<ECLIPSE_NAME>-%d",  # model, ecl
        ConfigKeys.ENSPATH: "../output/storage/<CASE_DIR>",  # model
        "PLOT_PATH": "../output/results/plot/<CASE_DIR>",
        ConfigKeys.UPDATE_LOG_PATH: "../output/update_log/<CASE_DIR>",  # analysis
        ConfigKeys.RUNPATH_FILE: (
            "../output/run_path_file/.ert-runpath-list_<CASE_DIR>"
        ),  # subst
        ConfigKeys.DEFINE_KEY: {
            "<USER>": "TEST_USER",
            "<SCRATCH>": "scratch/ert",
            "<CASE_DIR>": "the_extensive_case",
            "<ECLIPSE_NAME>": "XYZ",
        },  # subst
        ConfigKeys.REFCASE: "../input/refcase/SNAKE_OIL_FIELD",  # ecl
        ConfigKeys.JOBNAME: "SNAKE_OIL_STRUCTURE_%d",  # model
        ConfigKeys.MAX_RESAMPLE: 1,  # model
        ConfigKeys.TIME_MAP: "../input/refcase/time_map.txt",  # model
        ConfigKeys.INSTALL_JOB: [
            {
                ConfigKeys.NAME: "SNAKE_OIL_SIMULATOR",
                ConfigKeys.PATH: "../../snake_oil/jobs/SNAKE_OIL_SIMULATOR",
            },
            {
                ConfigKeys.NAME: "SNAKE_OIL_NPV",
                ConfigKeys.PATH: "../../snake_oil/jobs/SNAKE_OIL_NPV",
            },
            {
                ConfigKeys.NAME: "SNAKE_OIL_DIFF",
                ConfigKeys.PATH: "../../snake_oil/jobs/SNAKE_OIL_DIFF",
            },  # site
        ],
        ConfigKeys.FORWARD_MODEL: [
            {
                ConfigKeys.NAME: "SNAKE_OIL_SIMULATOR",
                ConfigKeys.ARGLIST: "",
            },
            {
                ConfigKeys.NAME: "SNAKE_OIL_NPV",
                ConfigKeys.ARGLIST: "",
            },
            {ConfigKeys.NAME: "SNAKE_OIL_DIFF", ConfigKeys.ARGLIST: ""},  # model
        ],
        ConfigKeys.HISTORY_SOURCE: HistorySourceEnum.REFCASE_HISTORY,
        ConfigKeys.OBS_CONFIG: "../input/observations/obsfiles/observations.txt",
        ConfigKeys.GEN_KW_EXPORT_NAME: "parameters",
        ConfigKeys.LOAD_WORKFLOW_JOB: [
            {
                ConfigKeys.NAME: "UBER_PRINT",
                ConfigKeys.PATH: "../bin/workflows/workflowjobs/UBER_PRINT",
            }  # workflow_list
        ],
        ConfigKeys.LOAD_WORKFLOW: [
            {
                ConfigKeys.NAME: "MAGIC_PRINT",
                ConfigKeys.PATH: "../bin/workflows/MAGIC_PRINT",
            }  # workflow_list
        ],
        "RNG_ALG_TYPE": RngAlgTypeEnum.MZRAN,
        ConfigKeys.RANDOM_SEED: "3593114179000630026631423308983283277868",  # rng
        ConfigKeys.GRID: "../../eclipse/include/grid/CASE.EGRID",  # ecl
        ConfigKeys.RUN_TEMPLATE: [
            (
                "../input/templates/seed_template.txt",
                "seed.txt",
            )  # ert_templates not sure about this we might do a proper dict instead?
        ],
    }

    # replace define keys only in root strings, this should be updated
    # and validated in configsuite instead
    for define_key in config_data_new[ConfigKeys.DEFINE_KEY]:
        for data_key, data_value in config_data_new.items():
            if isinstance(data_value, str):
                config_data_new[data_key] = data_value.replace(
                    define_key,
                    config_data_new[ConfigKeys.DEFINE_KEY].get(define_key),
                )

    # change dir to actual location of cfg_file
    os.chdir(cfg_path)

    # add missing entries to config file
    with open(cfg_file, "a+") as ert_file:
        ert_file.write("JOB_SCRIPT ../../../script.sh\n")
        ert_file.write("NUM_CPU 0\n")

    # load res_file
    res_config_file = ResConfig(user_config_file=cfg_file)

    # get site_config location
    ert_share_path = os.path.dirname(res_config_file.site_config.getLocation())

    # update dictionary
    # commit missing entries, this should be updated and validated in
    # configsuite instead
    config_data_new[ConfigKeys.CONFIG_FILE_KEY] = cfg_file
    config_data_new[ConfigKeys.INSTALL_JOB_DIRECTORY] = [
        ert_share_path + "/forward-models/res",
        ert_share_path + "/forward-models/shell",
        ert_share_path + "/forward-models/templating",
        ert_share_path + "/forward-models/old_style",
    ]
    config_data_new[ConfigKeys.WORKFLOW_JOB_DIRECTORY] = [
        ert_share_path + "/workflows/jobs/shell",
        ert_share_path + "/workflows/jobs/internal-gui/config",
    ]
    for ip in config_data_new[ConfigKeys.INSTALL_JOB]:
        ip[ConfigKeys.PATH] = os.path.realpath(ip[ConfigKeys.PATH])

    for ip in config_data_new[ConfigKeys.LOAD_WORKFLOW]:
        ip[ConfigKeys.PATH] = os.path.realpath(ip[ConfigKeys.PATH])
    for ip in config_data_new[ConfigKeys.LOAD_WORKFLOW_JOB]:
        ip[ConfigKeys.PATH] = os.path.realpath(ip[ConfigKeys.PATH])

    config_data_new[ConfigKeys.JOB_SCRIPT] = os.path.normpath(
        os.path.realpath(config_data_new[ConfigKeys.JOB_SCRIPT])
    )

    # open config via dictionary
    res_config_dict = ResConfig(config_dict=config_data_new)

    assert res_config_file.subst_config == res_config_dict.subst_config
    assert res_config_file.site_config == res_config_dict.site_config
    assert res_config_file.rng_config == res_config_dict.rng_config
    assert res_config_file.ert_workflow_list == res_config_dict.ert_workflow_list
    assert res_config_file.hook_manager == res_config_dict.hook_manager
    assert res_config_file.ert_templates == res_config_dict.ert_templates
    assert res_config_file.ecl_config == res_config_dict.ecl_config
    assert res_config_file.ensemble_config == res_config_dict.ensemble_config
    assert res_config_file.model_config == res_config_dict.model_config
    # https://github.com/equinor/ert/issues/2571
    # assert res_config_file.queue_config == res_config_dict.queue_config


def test_runpath_file(monkeypatch, tmp_path):
    """
    There was an issue relating to `ResConfig.runpath_file` returning a
    relative path rather than an absolute path. This test simulates the
    conditions that caused the original bug. That is, the user starts
    somewhere else and points to the ERT config file using a relative
    path.
    """
    config_path = tmp_path / "model/ert/config.ert"
    workdir_path = tmp_path / "start/from/here"
    runpath_path = tmp_path / "model/output/my_custom_runpath_path.foo"

    config_path.parent.mkdir(parents=True)
    workdir_path.mkdir(parents=True)
    monkeypatch.chdir(workdir_path)

    with config_path.open("w") as f:
        f.writelines(
            [
                "DEFINE <FOO> foo\n",
                "RUNPATH_FILE ../output/my_custom_runpath_path.<FOO>\n",
                # Required for this to be a valid ResConfig
                "NUM_REALIZATIONS 1\n",
            ]
        )

    config = ResConfig(os.path.relpath(config_path, workdir_path))
    assert config.runpath_file == str(runpath_path)
