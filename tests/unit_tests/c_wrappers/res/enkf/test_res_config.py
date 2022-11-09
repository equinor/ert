import logging
import os
import os.path
import stat
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from cwrap import Prototype, load
from ecl.util.enums import RngAlgTypeEnum

from ert._c_wrappers.enkf import (
    AnalysisConfig,
    ConfigKeys,
    GenDataFileType,
    ResConfig,
    SiteConfig,
)
from ert._c_wrappers.enkf.enums import HookRuntime
from ert._c_wrappers.enkf.res_config import parse_signature_job, site_config_location
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


def expand_config_defs(defines, config):
    for define_key, define_value in defines.items():
        for data_key, data_value in config.items():
            if isinstance(data_value, str):
                config[data_key] = data_value.replace(define_key, define_value)


# Expand all strings in snake oil structure config according to defines.
expand_config_defs(config_defines, snake_oil_structure_config)


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

    assert res_config.substitution_list["<CONFIG_PATH>"] == os.getcwd()


def test_extensive_config(setup_case):
    res_config = setup_case("snake_oil_structure", "ert/model/user_config.ert")

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
        == res_config.forward_model.job_name_list()
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
    assert queue_config.queue_system == QueueDriverEnum.LSF_DRIVER
    assert snake_oil_structure_config["MAX_SUBMIT"] == queue_config.max_submit
    driver = queue_config.create_driver()
    assert snake_oil_structure_config["MAX_RUNNING"] == driver.get_option("MAX_RUNNING")
    assert snake_oil_structure_config["LSF_SERVER"] == driver.get_option("LSF_SERVER")
    assert snake_oil_structure_config["LSF_RESOURCE"] == driver.get_option(
        "LSF_RESOURCE"
    )

    site_config = res_config.site_config
    job_list = site_config.job_list
    for job_name in snake_oil_structure_config["INSTALL_JOB"]:
        assert job_name in job_list

        exp_job_data = snake_oil_structure_config["INSTALL_JOB"][job_name]

        assert (
            Path(exp_job_data["CONFIG"]).resolve()
            == Path(job_list[job_name].get_config_file()).resolve()
        )
        assert exp_job_data["STDERR"] == job_list[job_name].get_stderr_file()
        assert exp_job_data["STDOUT"] == job_list[job_name].get_stdout_file()

    ensemble_config = res_config.ensemble_config
    for extension in ["SMSPEC", "UNSMRY"]:
        assert (
            Path(snake_oil_structure_config["REFCASE"] + "." + extension).resolve()
            == Path(ensemble_config.refcase.case + "." + extension).resolve()
        )
    assert (
        Path(snake_oil_structure_config["GRID"]).resolve()
        == Path(ensemble_config._grid_file).resolve()
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

    assert snake_oil_structure_config["RNG_ALG_TYPE"] == RngAlgTypeEnum.MZRAN


def test_res_config_dict_constructor(setup_case):
    config_file_name = "user_config.ert"
    relative_config_path = f"ert/model/{config_file_name}"
    _ = setup_case("snake_oil_structure", relative_config_path)
    # create script file
    script_file = "script.sh"
    with open(file=script_file, mode="w", encoding="utf-8") as f:
        f.write("""#!/bin/sh\nls""")

    st = os.stat(script_file)
    os.chmod(script_file, stat.S_IEXEC | st.st_mode)

    # split config_file to path and filename
    absolute_config_dir, _ = os.path.split(os.path.realpath(relative_config_path))

    config_data_new = {
        ConfigKeys.ALPHA_KEY: 3,
        ConfigKeys.RERUN_KEY: False,
        ConfigKeys.RERUN_START_KEY: 0,
        ConfigKeys.STD_CUTOFF_KEY: 1e-6,
        ConfigKeys.STOP_LONG_RUNNING: False,
        ConfigKeys.GLOBAL_STD_SCALING: 1,
        ConfigKeys.MIN_REALIZATIONS: 5,
        # "MIN_REALIZATIONS"  : "50%", percentages need to be fixed or removed
        ConfigKeys.RUNPATH: "<SCRATCH>/<USER>/<CASE_DIR>/realization-%d/iter-%d",
        ConfigKeys.NUM_REALIZATIONS: 10,  # model
        ConfigKeys.MAX_RUNTIME: 23400,
        ConfigKeys.JOB_SCRIPT: f"../../{script_file}",
        ConfigKeys.QUEUE_SYSTEM: QueueDriverEnum.LSF_DRIVER,
        ConfigKeys.MAX_SUBMIT: 13,
        ConfigKeys.QUEUE_OPTION: [
            {
                ConfigKeys.DRIVER_NAME: QueueDriverEnum.LSF_DRIVER,
                ConfigKeys.OPTION: "MAX_RUNNING",
                ConfigKeys.VALUE: "100",
            },
            {
                ConfigKeys.DRIVER_NAME: QueueDriverEnum.LSF_DRIVER,
                ConfigKeys.OPTION: "LSF_QUEUE",
                ConfigKeys.VALUE: "mr",
            },
            {
                ConfigKeys.DRIVER_NAME: QueueDriverEnum.LSF_DRIVER,
                ConfigKeys.OPTION: "LSF_SERVER",
                ConfigKeys.VALUE: "simulacrum",
            },
            {
                ConfigKeys.DRIVER_NAME: QueueDriverEnum.LSF_DRIVER,
                ConfigKeys.OPTION: "LSF_RESOURCE",
                ConfigKeys.VALUE: "select[x86_64Linux] same[type:model]",
            },
        ],
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
                ConfigKeys.FORWARD_INIT: False,
            }  # ensemble
        ],
        ConfigKeys.GEN_DATA: [
            {
                ConfigKeys.NAME: "super_data",
                ConfigKeys.INPUT_FORMAT: GenDataFileType.ASCII,
                ConfigKeys.RESULT_FILE: "super_data_%d",
                ConfigKeys.REPORT_STEPS: [1],
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
            "<CWD>": absolute_config_dir,
            "<CONFIG_PATH>": absolute_config_dir,
            "<CONFIG_FILE>": config_file_name,
            "<CONFIG_FILE_BASE>": config_file_name.split(".", maxsplit=1)[0],
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

    # change dir to actual location of config file
    os.chdir(absolute_config_dir)

    # add missing entries to config file
    with open(file=config_file_name, mode="a+", encoding="utf-8") as ert_file:
        ert_file.write(f"JOB_SCRIPT ../../{script_file}\n")

    # load res_file
    res_config_file = ResConfig(user_config_file=config_file_name)

    # get site_config location
    ert_share_path = os.path.dirname(site_config_location())

    # update dictionary
    # commit missing entries, this should be updated and validated in
    # configsuite instead
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

    assert res_config_file.substitution_list == res_config_dict.substitution_list
    assert res_config_file.site_config == res_config_dict.site_config
    assert res_config_file.random_seed == res_config_dict.random_seed
    assert res_config_file.ert_workflow_list == res_config_dict.ert_workflow_list
    assert res_config_file.ert_templates == res_config_dict.ert_templates
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


def test_that_job_script_can_be_set_in_site_config(monkeypatch, tmp_path):
    """
    We use the jobscript field to inject a komodo environment onprem.
    This overwrites the value by appending to the default siteconfig.
    Need to check that the second JOB_SCRIPT is the one that gets used.
    """
    test_site_config = tmp_path / "test_site_config.ert"
    my_script = (tmp_path / "my_script").resolve()
    my_script.write_text("")
    st = os.stat(my_script)
    os.chmod(my_script, st.st_mode | stat.S_IEXEC)
    test_site_config.write_text(
        f"JOB_SCRIPT job_dispatch.py\nJOB_SCRIPT {my_script}\nQUEUE_SYSTEM LOCAL\n"
    )
    monkeypatch.setenv("ERT_SITE_CONFIG", str(test_site_config))

    test_user_config = tmp_path / "user_config.ert"

    test_user_config.write_text(
        "JOBNAME  Job%d\nRUNPATH /tmp/simulations/run%d\nNUM_REALIZATIONS 10\n"
    )

    res_config = ResConfig(str(test_user_config))

    assert Path(res_config.queue_config.job_script).resolve() == my_script


def test_that_unknown_queue_option_gives_error_message(
    caplog, monkeypatch, tmp_path, capsys
):
    test_site_config = tmp_path / "test_site_config.ert"
    my_script = (tmp_path / "my_script").resolve()
    my_script.write_text("")
    st = os.stat(my_script)
    os.chmod(my_script, st.st_mode | stat.S_IEXEC)
    test_site_config.write_text(
        f"JOB_SCRIPT job_dispatch.py\nJOB_SCRIPT {my_script}\nQUEUE_SYSTEM LOCAL\n"
    )
    monkeypatch.setenv("ERT_SITE_CONFIG", str(test_site_config))

    test_user_config = tmp_path / "user_config.ert"

    test_user_config.write_text(
        "JOBNAME  Job%d\nRUNPATH /tmp/simulations/run%d\n"
        "NUM_REALIZATIONS 10\nQUEUE_OPTION UNKNOWN_QUEUE unsetoption\n"
    )

    with pytest.raises(ValueError, match="Parsing"):
        _ = ResConfig(str(test_user_config))

    err = capsys.readouterr().err
    assert "Errors parsing" in err
    assert "UNKNOWN_QUEUE" in err


@pytest.mark.parametrize(
    "run_mode",
    [
        HookRuntime.POST_SIMULATION,
        HookRuntime.PRE_SIMULATION,
        HookRuntime.PRE_FIRST_UPDATE,
        HookRuntime.PRE_UPDATE,
        HookRuntime.POST_UPDATE,
    ],
)
def test_that_workflow_run_modes_can_be_selected(tmp_path, run_mode):
    my_script = (tmp_path / "my_script").resolve()
    my_script.write_text("")
    st = os.stat(my_script)
    os.chmod(my_script, st.st_mode | stat.S_IEXEC)
    test_user_config = tmp_path / "user_config.ert"
    test_user_config.write_text(
        "JOBNAME  Job%d\nRUNPATH /tmp/simulations/run%d\n"
        "NUM_REALIZATIONS 10\n"
        f"LOAD_WORKFLOW {my_script} SCRIPT\n"
        f"HOOK_WORKFLOW SCRIPT {run_mode}\n"
    )
    res_config = ResConfig(str(test_user_config))
    assert (
        len(list(res_config.ert_workflow_list.get_workflows_hooked_at(run_mode))) == 1
    )


@pytest.mark.parametrize(
    "config_content, expected",
    [
        pytest.param("--Comment", "", id="Line comment"),
        pytest.param(" --Comment", "", id="Line comment with whitespace"),
        pytest.param("\t--Comment", "", id="Line comment with whitespace"),
        pytest.param("KEY VALUE", "KEY VALUE\n", id="Config line"),
        pytest.param("KEY VALUE --Comment", "KEY VALUE\n", id="Inline comment"),
    ],
)
def test_logging_config(caplog, config_content, expected):
    base_content = "Content of the configuration file (file_name):\n{}"
    config_path = "file_name"

    with patch("builtins.open", mock_open(read_data=config_content)), patch(
        "os.path.isfile", MagicMock(return_value=True)
    ):
        with caplog.at_level(logging.INFO):
            with patch.object(ResConfig, "__init__", lambda x: None):
                res_config = ResConfig()
                res_config._log_config_file(config_path)
    expected = base_content.format(expected)
    assert expected in caplog.messages


def test_logging_snake_oil_config(caplog, source_root):
    """
    Run logging on an actual config file with line comments
    and inline comments to check the result
    """

    config_path = os.path.join(
        source_root,
        "test-data",
        "snake_oil_structure",
        "ert",
        "model",
        "user_config.ert",
    )
    with caplog.at_level(logging.INFO), patch.object(
        ResConfig, "__init__", lambda x: None
    ):
        res_config = ResConfig()
        res_config._log_config_file(config_path)
    assert (
        """
JOBNAME SNAKE_OIL_STRUCTURE_%d
DEFINE  <USER>          TEST_USER
DEFINE  <SCRATCH>       scratch/ert
DEFINE  <CASE_DIR>      the_extensive_case
DEFINE  <ECLIPSE_NAME>  XYZ
DATA_FILE           ../../eclipse/model/SNAKE_OIL.DATA
GRID                ../../eclipse/include/grid/CASE.EGRID
RUNPATH             <SCRATCH>/<USER>/<CASE_DIR>/realization-%d/iter-%d
ECLBASE             eclipse/model/<ECLIPSE_NAME>-%d
ENSPATH             ../output/storage/<CASE_DIR>
RUNPATH_FILE        ../output/run_path_file/.ert-runpath-list_<CASE_DIR>
REFCASE             ../input/refcase/SNAKE_OIL_FIELD
UPDATE_LOG_PATH     ../output/update_log/<CASE_DIR>
RANDOM_SEED 3593114179000630026631423308983283277868
NUM_REALIZATIONS              10
MAX_RUNTIME                   23400
MIN_REALIZATIONS              50%
QUEUE_SYSTEM                  LSF
QUEUE_OPTION LSF MAX_RUNNING  100
QUEUE_OPTION LSF LSF_RESOURCE select[x86_64Linux] same[type:model]
QUEUE_OPTION LSF LSF_SERVER   simulacrum
QUEUE_OPTION LSF LSF_QUEUE    mr
MAX_SUBMIT                    13
GEN_DATA super_data INPUT_FORMAT:ASCII RESULT_FILE:super_data_%d  REPORT_STEPS:1
GEN_KW SIGMA          ../input/templates/sigma.tmpl          coarse.sigma              ../input/distributions/sigma.dist
RUN_TEMPLATE             ../input/templates/seed_template.txt     seed.txt
INSTALL_JOB SNAKE_OIL_SIMULATOR ../../snake_oil/jobs/SNAKE_OIL_SIMULATOR
INSTALL_JOB SNAKE_OIL_NPV ../../snake_oil/jobs/SNAKE_OIL_NPV
INSTALL_JOB SNAKE_OIL_DIFF ../../snake_oil/jobs/SNAKE_OIL_DIFF
HISTORY_SOURCE REFCASE_HISTORY
OBS_CONFIG ../input/observations/obsfiles/observations.txt
TIME_MAP   ../input/refcase/time_map.txt
SUMMARY WOPR:PROD
SUMMARY WOPT:PROD
SUMMARY WWPR:PROD
SUMMARY WWCT:PROD
SUMMARY WWPT:PROD
SUMMARY WBHP:PROD
SUMMARY WWIR:INJ
SUMMARY WWIT:INJ
SUMMARY WBHP:INJ
SUMMARY ROE:1"""  # noqa: E501 pylint: disable=line-too-long
        in caplog.text
    )


def test_that_parse_job_signature_passes_through_job_names():
    assert parse_signature_job("JOB") == ("JOB", None)


def test_that_parse_job_signature_correctly_gets_arguments():
    assert parse_signature_job("JOB(<ARG1>=val1, <ARG2>=val2)") == (
        "JOB",
        "<ARG1>=val1, <ARG2>=val2",
    )


def test_that_parse_job_signature_warns_for_extra_parens(caplog):
    assert parse_signature_job("JOB(<ARG1>=val1, <ARG2>=val2), <ARG3>=val3)") == (
        "JOB",
        "<ARG1>=val1, <ARG2>=val2",
    )
    assert "Arguments after closing )" in caplog.text
