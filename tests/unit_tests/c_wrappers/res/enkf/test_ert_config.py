import logging
import os
import os.path
import stat
from datetime import date
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, mock_open, patch

import pytest
from ecl.util.enums import RngAlgTypeEnum

from ert._c_wrappers.enkf import AnalysisConfig, ConfigKeys, ErtConfig, HookRuntime
from ert._c_wrappers.enkf.ert_config import site_config_location
from ert._c_wrappers.sched import HistorySourceEnum
from ert.job_queue import QueueDriverEnum
from ert.parsing import ConfigValidationError

config_defines = {
    "<USER>": "TEST_USER",
    "<SCRATCH>": "ert/model/scratch/ert",
    "<CASE_DIR>": "the_extensive_case",
    "<ECLIPSE_NAME>": "XYZ",
}

snake_oil_structure_config = {
    "RUNPATH": "<SCRATCH>/<USER>/<CASE_DIR>/realization-<IENS>/iter-<ITER>",
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
    "ECLBASE": "eclipse/model/<ECLIPSE_NAME>-<IENS>",
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
    "JOBNAME": "SNAKE_OIL_STRUCTURE_<IENS>",
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
    "HISTORY_SOURCE": "REFCASE_HISTORY",
    "OBS_CONFIG": "ert/input/observations/obsfiles/observations.txt",
    "LOAD_WORKFLOW": [["ert/bin/workflows/MAGIC_PRINT", "MAGIC_PRINT"]],
    "LOAD_WORKFLOW_JOB": [
        ["ert/bin/workflows/workflowjobs/bin/uber_print.py", "UBER_PRINT"],
    ],
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
        ErtConfig.from_file("this/is/not/a/file")


def test_include_non_existing_file(tmpdir):
    with tmpdir.as_cwd():
        config = """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        INCLUDE does_not_exists
        """
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)

        with pytest.raises(
            ConfigValidationError, match=r"INCLUDE file:.*does_not_exists not found"
        ):
            _ = ErtConfig.from_file("config.ert")


def test_include_existing_file(tmpdir):
    with tmpdir.as_cwd():
        config = """
        JOBNAME my_name%d
        INCLUDE include_me
        NUM_REALIZATIONS 1
        """
        rand_seed = 420
        include_me_text = f"""
        RANDOM_SEED {rand_seed}
        """

        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)

        with open("include_me", mode="w", encoding="utf-8") as fh:
            fh.writelines(include_me_text)

        ert_config = ErtConfig.from_file("config.ert")
        assert ert_config.random_seed == str(rand_seed)


def test_init(minimum_case):
    ert_config = minimum_case.resConfig()

    assert ert_config is not None

    assert ert_config.analysis_config is not None
    assert isinstance(ert_config.analysis_config, AnalysisConfig)

    assert ert_config.config_path == os.getcwd()

    assert ert_config.substitution_list["<CONFIG_PATH>"] == os.getcwd()


def test_extensive_config(setup_case):
    ert_config = setup_case("snake_oil_structure", "ert/model/user_config.ert")

    assert (
        Path(snake_oil_structure_config["ENSPATH"]).resolve()
        == Path(ert_config.ens_path).resolve()
    )
    model_config = ert_config.model_config
    assert (
        Path(snake_oil_structure_config["RUNPATH"]).resolve()
        == Path(model_config.runpath_format_string).resolve()
    )
    jobname_format = snake_oil_structure_config["JOBNAME"].replace("%d", "<IENS>")
    assert jobname_format == model_config.jobname_format_string
    assert (
        snake_oil_structure_config["FORWARD_MODEL"]
        == ert_config.forward_model_job_name_list()
    )
    assert (
        HistorySourceEnum.from_string(snake_oil_structure_config["HISTORY_SOURCE"])
        == model_config.history_source
    )
    assert (
        snake_oil_structure_config["NUM_REALIZATIONS"] == model_config.num_realizations
    )
    assert (
        Path(snake_oil_structure_config["OBS_CONFIG"]).resolve()
        == Path(model_config.obs_config_file).resolve()
    )

    analysis_config = ert_config.analysis_config
    assert (
        snake_oil_structure_config["MAX_RUNTIME"] == analysis_config.get_max_runtime()
    )
    assert (
        Path(snake_oil_structure_config["UPDATE_LOG_PATH"]).resolve()
        == Path(analysis_config.get_log_path()).resolve()
    )

    queue_config = ert_config.queue_config
    assert queue_config.queue_system == QueueDriverEnum.LSF_DRIVER
    assert snake_oil_structure_config["MAX_SUBMIT"] == queue_config.max_submit
    driver = queue_config.create_driver()
    assert snake_oil_structure_config["MAX_RUNNING"] == driver.get_option("MAX_RUNNING")
    assert snake_oil_structure_config["LSF_SERVER"] == driver.get_option("LSF_SERVER")
    assert snake_oil_structure_config["LSF_RESOURCE"] == driver.get_option(
        "LSF_RESOURCE"
    )

    for job_name in snake_oil_structure_config["INSTALL_JOB"]:
        job = ert_config.installed_jobs[job_name]

        exp_job_data = snake_oil_structure_config["INSTALL_JOB"][job_name]

        assert exp_job_data["STDERR"] == job.stderr_file
        assert exp_job_data["STDOUT"] == job.stdout_file

    ensemble_config = ert_config.ensemble_config
    for extension in ["SMSPEC", "UNSMRY"]:
        assert (
            Path(snake_oil_structure_config["REFCASE"] + "." + extension).resolve()
            == Path(ensemble_config.refcase.case + "." + extension).resolve()
        )
    assert (
        Path(snake_oil_structure_config["GRID"]).resolve()
        == Path(ensemble_config._grid_file).resolve()
    )

    ensemble_config = ert_config.ensemble_config
    assert set(
        ["summary"]
        + snake_oil_structure_config["GEN_KW"]
        + snake_oil_structure_config["GEN_DATA"]
    ) == set(ensemble_config.keys)

    assert (
        Path(snake_oil_structure_config["SIGMA"]["TEMPLATE"]).resolve()
        == Path(ensemble_config.getNode("SIGMA").template_file).resolve()
    )
    assert (
        Path(snake_oil_structure_config["SIGMA"]["PARAMETER"]).resolve()
        == Path(ensemble_config.getNode("SIGMA").parameter_file).resolve()
    )
    assert (
        Path(snake_oil_structure_config["SIGMA"]["RESULT"]).resolve()
        == Path(ensemble_config["SIGMA"].output_file).resolve()
    )

    assert len(snake_oil_structure_config["LOAD_WORKFLOW"]) == len(
        list(ert_config.workflows.keys())
    )

    for w_path, w_name in snake_oil_structure_config["LOAD_WORKFLOW"]:
        assert w_name in ert_config.workflows
        assert (
            Path(w_path).resolve()
            == Path(ert_config.workflows[w_name].src_file).resolve()
        )

    for wj_path, wj_name in snake_oil_structure_config["LOAD_WORKFLOW_JOB"]:
        assert wj_name in ert_config.workflow_jobs
        job = ert_config.workflow_jobs[wj_name]

        assert wj_name == job.name
        assert Path(wj_path).resolve() == Path(job.executable).resolve()

    assert snake_oil_structure_config["RNG_ALG_TYPE"] == RngAlgTypeEnum.MZRAN


def test_runpath_file(monkeypatch, tmp_path):
    """
    There was an issue relating to `ErtConfig.runpath_file` returning a
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
                # Required for this to be a valid ErtConfig
                "NUM_REALIZATIONS 1\n",
            ]
        )

    config = ErtConfig.from_file(os.path.relpath(config_path, workdir_path))
    assert config.runpath_file == runpath_path


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
        "JOBNAME  Job%d\nRUNPATH /tmp/simulations/realization-<IENS>/iter-<ITER>\n"
        "NUM_REALIZATIONS 10\n"
    )

    ert_config = ErtConfig.from_file(str(test_user_config))

    assert Path(ert_config.queue_config.job_script).resolve() == my_script


def test_that_unknown_queue_option_gives_error_message(monkeypatch, tmp_path):
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
        "JOBNAME  Job%d\nRUNPATH /tmp/simulations/realization-<IENS>/iter-<ITER>\n"
        "NUM_REALIZATIONS 10\nQUEUE_OPTION UNKNOWN_QUEUE unsetoption\n"
    )

    with pytest.raises(
        ConfigValidationError, match="'QUEUE_OPTION' argument 0 must be one of"
    ):
        _ = ErtConfig.from_file(str(test_user_config))


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
        "JOBNAME  Job%d\nRUNPATH /tmp/simulations/realization-<IENS>/iter-<ITER>\n"
        "NUM_REALIZATIONS 10\n"
        f"LOAD_WORKFLOW {my_script} SCRIPT\n"
        f"HOOK_WORKFLOW SCRIPT {run_mode.name}\n"
    )
    ert_config = ErtConfig.from_file(str(test_user_config))
    assert len(list(ert_config.hooked_workflows[run_mode])) == 1


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
    ), caplog.at_level(logging.INFO):
        ErtConfig._log_config_file(config_path)
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
    with caplog.at_level(logging.INFO):
        ErtConfig._log_config_file(config_path)
    assert (
        """
JOBNAME SNAKE_OIL_STRUCTURE_%d
DEFINE  <USER>          TEST_USER
DEFINE  <SCRATCH>       scratch/ert
DEFINE  <CASE_DIR>      the_extensive_case
DEFINE  <ECLIPSE_NAME>  XYZ
DATA_FILE           ../../eclipse/model/SNAKE_OIL.DATA
GRID                ../../eclipse/include/grid/CASE.EGRID
RUNPATH             <SCRATCH>/<USER>/<CASE_DIR>/realization-<IENS>/iter-<ITER>
ECLBASE             eclipse/model/<ECLIPSE_NAME>-<IENS>
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


def test_that_parsing_workflows_gives_expected(use_tmpdir):
    ERT_SITE_CONFIG = site_config_location()
    ERT_SHARE_PATH = os.path.dirname(ERT_SITE_CONFIG)
    cwd = os.getcwd()

    config_dict = {
        ConfigKeys.LOAD_WORKFLOW_JOB: [
            [cwd + "/workflows/UBER_PRINT", "print_uber"],
            [cwd + "/workflows/HIDDEN_PRINT", "HIDDEN_PRINT"],
        ],
        ConfigKeys.LOAD_WORKFLOW: [
            [cwd + "/workflows/MAGIC_PRINT", "magic_print"],
            [cwd + "/workflows/NO_PRINT", "no_print"],
            [cwd + "/workflows/SOME_PRINT", "some_print"],
        ],
        ConfigKeys.WORKFLOW_JOB_DIRECTORY: [
            ERT_SHARE_PATH + "/workflows/jobs/shell",
            ERT_SHARE_PATH + "/workflows/jobs/internal/config",
            ERT_SHARE_PATH + "/workflows/jobs/internal-gui/config",
        ],
        ConfigKeys.HOOK_WORKFLOW_KEY: [
            ["magic_print", "POST_UPDATE"],
            ["no_print", "PRE_UPDATE"],
        ],
    }

    config_dict["ENSPATH"] = "enspath"
    config_dict["NUM_REALIZATIONS"] = 1

    with open("minimum_config", "a+", encoding="utf-8") as ert_file:
        ert_file.write("LOAD_WORKFLOW_JOB workflows/UBER_PRINT print_uber\n")
        ert_file.write("LOAD_WORKFLOW_JOB workflows/HIDDEN_PRINT\n")
        ert_file.write("LOAD_WORKFLOW workflows/MAGIC_PRINT magic_print\n")
        ert_file.write("LOAD_WORKFLOW workflows/NO_PRINT no_print\n")
        ert_file.write("LOAD_WORKFLOW workflows/SOME_PRINT some_print\n")
        ert_file.write("HOOK_WORKFLOW magic_print POST_UPDATE\n")
        ert_file.write("HOOK_WORKFLOW no_print PRE_UPDATE\n")

    os.mkdir("workflows")

    with open("workflows/MAGIC_PRINT", "w", encoding="utf-8") as f:
        f.write("print_uber\n")
    with open("workflows/NO_PRINT", "w", encoding="utf-8") as f:
        f.write("print_uber\n")
    with open("workflows/SOME_PRINT", "w", encoding="utf-8") as f:
        f.write("print_uber\n")
    with open("workflows/UBER_PRINT", "w", encoding="utf-8") as f:
        f.write("EXECUTABLE ls\n")
    with open("workflows/HIDDEN_PRINT", "w", encoding="utf-8") as f:
        f.write("EXECUTABLE ls\n")

    ert_config = ErtConfig.from_dict(config_dict)

    # verify name generated from filename
    assert "HIDDEN_PRINT" in ert_config.workflow_jobs
    assert "print_uber" in ert_config.workflow_jobs

    assert [
        "magic_print",
        "no_print",
        "some_print",
    ] == list(ert_config.workflows.keys())

    assert len(ert_config.hooked_workflows[HookRuntime.PRE_UPDATE]) == 1
    assert len(ert_config.hooked_workflows[HookRuntime.POST_UPDATE]) == 1
    assert len(ert_config.hooked_workflows[HookRuntime.PRE_FIRST_UPDATE]) == 0


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_plugin_jobs_fetches_exactly_ert_plugins():
    script_file_contents = dedent(
        """
        SCRIPT script.py
        INTERNAL True
        """
    )
    plugin_file_contents = dedent(
        """
        SCRIPT plugin.py
        INTERNAL True
        """
    )

    script_file_path = os.path.join(os.getcwd(), "script")
    plugin_file_path = os.path.join(os.getcwd(), "plugin")
    with open(script_file_path, mode="w", encoding="utf-8") as fh:
        fh.write(script_file_contents)
    with open(plugin_file_path, mode="w", encoding="utf-8") as fh:
        fh.write(plugin_file_contents)

    with open("script.py", mode="w", encoding="utf-8") as fh:
        fh.write(
            dedent(
                """
                from ert import ErtScript
                class Script(ErtScript):
                    def run(self, *args):
                        pass
                """
            )
        )
    with open("plugin.py", mode="w", encoding="utf-8") as fh:
        fh.write(
            dedent(
                """
                from ert.job_queue import ErtPlugin
                class Plugin(ErtPlugin):
                    def run(self, *args):
                        pass
                """
            )
        )
    with open("config.ert", mode="w", encoding="utf-8") as fh:
        fh.write(
            dedent(
                f"""
                NUM_REALIZATIONS 1
                LOAD_WORKFLOW_JOB {plugin_file_path} plugin
                LOAD_WORKFLOW_JOB {script_file_path} script
                """
            )
        )

    ert_config = ErtConfig.from_file("config.ert")

    assert ert_config.workflow_jobs["plugin"].is_plugin()
    assert not ert_config.workflow_jobs["script"].is_plugin()


def test_data_file_with_non_utf_8_character_gives_error_message(tmpdir):
    with tmpdir.as_cwd():
        data_file = "data_file.DATA"
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.write(
                """NUM_REALIZATIONS 1
            DATA_FILE data_file.DATA
            ECLBASE data_file_<ITER>
            """
            )
        with open(data_file, mode="w", encoding="utf-8") as fh:
            fh.write(
                dedent(
                    """
                        START
                        --  DAY   MONTH  YEAR
                        1    'JAN'  2017   /
                    """
                )
            )
        with open(data_file, "ab") as f:
            f.write(b"\xff")
        data_file_path = f"{tmpdir}/{data_file}"
        with pytest.raises(
            ConfigValidationError,
            match="Unsupported non UTF-8 character "
            f"'ÿ' found in file: {data_file_path!r}",
        ):
            ErtConfig.from_file("config.ert")
