import fileinput
import json
import logging
import os
import os.path
import stat
import warnings
from datetime import date
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock

import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from pydantic import RootModel, TypeAdapter

from ert import ErtScript, ErtScriptWorkflow
from ert.config import (
    ConfigValidationError,
    ErtConfig,
    ESSettings,
    HookRuntime,
    QueueSystem,
)
from ert.config.ert_config import _split_string_into_sections, create_forward_model_json
from ert.config.forward_model_step import ForwardModelStep
from ert.config.parsing import ConfigKeys, ConfigWarning
from ert.config.parsing.context_values import (
    ContextBool,
    ContextBoolEncoder,
    ContextFloat,
    ContextInt,
    ContextList,
    ContextString,
)
from ert.plugins import ErtPluginManager
from ert.shared import ert_share_path
from ert.storage import LocalEnsemble, Storage
from tests.ert.conftest import _create_design_matrix

from .config_dict_generator import config_generators


@pytest.mark.usefixtures("use_tmpdir")
def test_config_can_include_existing_file():
    config = dedent("""
        INCLUDE include_me
        NUM_REALIZATIONS 1
        """)
    rand_seed = 420
    include_me_text = f"RANDOM_SEED {rand_seed}\n"

    Path("config.ert").write_text(config, encoding="utf-8")
    Path("include_me").write_text(include_me_text, encoding="utf-8")
    ert_config = ErtConfig.from_file("config.ert")

    assert ert_config.random_seed == rand_seed


@pytest.mark.usefixtures("use_tmpdir")
def test_that_config_path_substitution_is_the_name_of_the_configs_directory():
    Path("minimal_config.ert").write_text("NUM_REALIZATIONS 1", encoding="utf-8")
    ert_config = ErtConfig.from_file("minimal_config.ert")

    assert ert_config.config_path == os.getcwd()
    assert ert_config.substitutions["<CONFIG_PATH>"] == os.getcwd()


def test_runpath_file_is_absolute(monkeypatch, tmp_path):
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

    config_path.write_text(
        dedent("""
        DEFINE <FOO> foo
        RUNPATH_FILE ../output/my_custom_runpath_path.<FOO>
        -- Required for this to be a valid ErtConfig
        NUM_REALIZATIONS 1
        """),
        encoding="utf-8",
    )

    config = ErtConfig.from_file(os.path.relpath(config_path, workdir_path))
    assert config.runpath_file == runpath_path


@pytest.mark.usefixtures("use_tmpdir")
def test_that_job_script_can_be_set_in_site_config(monkeypatch):
    """
    We use the jobscript field to inject a komodo environment onprem.
    This overwrites the value by appending to the default siteconfig.
    Need to check that the second JOB_SCRIPT is the one that gets used.
    """
    test_site_config = Path("test_site_config.ert")
    my_script = Path("my_script").resolve()
    my_script.write_text("", encoding="utf-8")
    st = os.stat(my_script)
    os.chmod(my_script, st.st_mode | stat.S_IEXEC)
    test_site_config.write_text(
        dedent(f"""
        JOB_SCRIPT fm_dispatch.py
        JOB_SCRIPT {my_script}
        QUEUE_SYSTEM LOCAL
        """),
        encoding="utf-8",
    )
    monkeypatch.setenv("ERT_SITE_CONFIG", str(test_site_config))

    test_user_config = Path("user_config.ert")

    test_user_config.write_text(
        dedent("""
        JOBNAME Job%d
        NUM_REALIZATIONS 1
        """),
        encoding="utf-8",
    )

    ert_config = ErtConfig.from_file(test_user_config)

    assert Path(ert_config.queue_config.queue_options.job_script).resolve() == my_script


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
@pytest.mark.usefixtures("use_tmpdir")
def test_that_workflow_run_modes_can_be_selected(run_mode):
    my_script = Path("my_script").resolve()
    my_script.write_text("", encoding="utf-8")
    st = os.stat(my_script)
    os.chmod(my_script, st.st_mode | stat.S_IEXEC)
    test_user_config = Path("user_config.ert")
    test_user_config.write_text(
        dedent(f"""JOBNAME Job%d
        NUM_REALIZATIONS 10
        LOAD_WORKFLOW {my_script} SCRIPT
        HOOK_WORKFLOW SCRIPT {run_mode.name}
        """),
        encoding="utf-8",
    )
    ert_config = ErtConfig.from_file(test_user_config)
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

    with caplog.at_level(logging.INFO):
        ErtConfig._log_config_file(config_path, config_content)
    expected = base_content.format(expected)
    assert expected in caplog.messages


@pytest.mark.usefixtures("use_tmpdir")
def test_custom_forward_models_are_logged(caplog):
    localhack = "localhack.sh"
    Path(localhack).write_text("", encoding="utf-8")
    st = os.stat(localhack)
    os.chmod(localhack, st.st_mode | stat.S_IEXEC)
    Path("foo_fm").write_text(
        f"-- A comment\n   \nEXECUTABLE {localhack}\n\n\n", encoding="utf-8"
    )
    Path("config.ert").write_text(
        "NUM_REALIZATIONS 1\nINSTALL_JOB foo_fm foo_fm", encoding="utf-8"
    )
    with caplog.at_level(logging.INFO):
        ErtConfig.from_file("config.ert")
    assert (
        "Custom forward_model_step foo_fm installed as: "
        f"-- A comment\nEXECUTABLE {localhack}" in caplog.messages
    )
    assert (
        sum("Custom forward_model_step" in logmessage for logmessage in caplog.messages)
        == 1
    ), "check if site-config fm were logged"


def test_logging_with_comments(caplog):
    """
    Run logging on an actual config file with line comments
    and inline comments to check the result
    """

    config = dedent(
        """
        NUM_REALIZATIONS 1 -- inline comment
        -- Regular comment
        ECLBASE PRED_RUN
        SUMMARY *
        """
    )
    with caplog.at_level(logging.INFO):
        ErtConfig._log_config_file("config.ert", config)
    assert (
        """
NUM_REALIZATIONS 1
ECLBASE PRED_RUN
SUMMARY *"""
        in caplog.text
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_parsing_workflows_gives_expected():
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
            ert_share_path() + "/workflows/jobs/shell",
        ],
        ConfigKeys.HOOK_WORKFLOW: [
            ["magic_print", "POST_UPDATE"],
            ["no_print", "PRE_UPDATE"],
        ],
        ConfigKeys.ENSPATH: "enspath",
        ConfigKeys.NUM_REALIZATIONS: 1,
    }

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

    assert list(ert_config.workflows.keys()) == [
        "magic_print",
        "no_print",
        "some_print",
    ]

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

    script_file_path = Path.cwd() / "script"
    plugin_file_path = Path.cwd() / "plugin"
    Path(script_file_path).write_text(script_file_contents, encoding="utf-8")
    Path(plugin_file_path).write_text(plugin_file_contents, encoding="utf-8")
    Path("script.py").write_text(
        dedent(
            """
            from ert import ErtScript

            class Script(ErtScript):
                def run(self, *args):
                    pass
            """
        ),
        encoding="utf-8",
    )
    Path("plugin.py").write_text(
        dedent(
            """
            from ert.plugins import ErtPlugin

            class Plugin(ErtPlugin):
                def run(self, *args):
                    pass
            """
        ),
        encoding="utf-8",
    )
    Path("config.ert").write_text(
        dedent(
            f"""
            NUM_REALIZATIONS 1
            LOAD_WORKFLOW_JOB {plugin_file_path} plugin
            LOAD_WORKFLOW_JOB {script_file_path} script
            """
        ),
        encoding="utf-8",
    )
    with pytest.warns(ConfigWarning, match="Deprecated keywords, SCRIPT and INTERNAL"):
        ert_config = ErtConfig.from_file("config.ert")

    assert ert_config.workflow_jobs["plugin"].is_plugin()
    assert not ert_config.workflow_jobs["script"].is_plugin()


@pytest.mark.usefixtures("use_tmpdir")
def test_data_file_with_non_utf_8_character_gives_error_message():
    data_file = "data_file.DATA"
    Path("config.ert").write_text(
        dedent("""
            NUM_REALIZATIONS 1
            DATA_FILE data_file.DATA
            ECLBASE data_file_<ITER>
            """),
        encoding="utf-8",
    )
    Path(data_file).write_text(
        dedent(
            """
            START
            --  DAY   MONTH  YEAR
            1    'JAN'  2017   /
            """
        ),
        encoding="utf-8",
    )
    with open(data_file, "ab") as f:
        f.write(b"\xff")
    data_file_path = str(Path.cwd() / data_file)
    with (
        pytest.raises(
            ConfigValidationError,
            match="Unsupported non UTF-8 character "
            f"'ÿ' found in file: {data_file_path!r}",
        ),
        pytest.warns(match="Failed to read NUM_CPU"),
    ):
        ErtConfig.from_file("config.ert")


def test_that_double_comments_are_handled():
    ert_config = ErtConfig.from_file_contents(
        """
        NUM_REALIZATIONS 1 -- foo -- bar -- 2
        JOBNAME &SUM$VAR@12@#£¤<
        """
    )
    assert ert_config.runpath_config.num_realizations == 1
    assert ert_config.runpath_config.jobname_format_string == "&SUM$VAR@12@#£¤<"


def test_that_strings_escape_comments():
    ert_config = ErtConfig.from_file_contents(
        """
        NUM_REALIZATIONS 1
        JOBNAME " -- "
        """
    )
    assert ert_config.runpath_config.jobname_format_string == " -- "


@pytest.mark.filterwarnings("ignore:.*Unknown keyword.*:ert.config.ConfigWarning")
def test_bad_user_config_file_error_message():
    with pytest.raises(ConfigValidationError, match="NUM_REALIZATIONS must be set"):
        _ = ErtConfig.from_file_contents("NUM_REL 10\n")


@pytest.mark.usefixtures("use_tmpdir")
def test_ert_config_parses_date():
    test_config_file_base = "test"
    test_config_file_name = f"{test_config_file_base}.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        DEFINE <STORAGE> storage/<CONFIG_FILE_BASE>-<DATE>
        RUNPATH <STORAGE>/runpath/realization-<IENS>/iter-<ITER>
        ENSPATH <STORAGE>/ensemble
        """
    )
    Path(test_config_file_name).write_text(test_config_contents, encoding="utf-8")
    ert_config = ErtConfig.from_file(test_config_file_name)

    date_string = date.today().isoformat()
    expected_storage = os.path.abspath(f"storage/{test_config_file_base}-{date_string}")
    expected_run_path = f"{expected_storage}/runpath/realization-<IENS>/iter-<ITER>"
    expected_ens_path = f"{expected_storage}/ensemble"
    assert ert_config.ens_path == expected_ens_path
    assert ert_config.runpath_config.runpath_format_string == expected_run_path


def test_that_subst_list_is_given_default_runpath_file():
    assert ErtConfig.from_file_contents("NUM_REALIZATIONS 1").substitutions[
        "<RUNPATH_FILE>"
    ] == os.path.abspath(ErtConfig.DEFAULT_RUNPATH_FILE)


@pytest.mark.integration_test
@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@pytest.mark.usefixtures("set_site_config")
@settings(max_examples=10)
@given(config_generators())
def test_that_creating_ert_config_from_dict_is_same_as_from_file(
    tmp_path_factory, config_generator
):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_values:
        config_from_dict = ErtConfig.from_dict(
            config_values.to_config_dict("config.ert", os.getcwd())
        )
        config_from_file = ErtConfig.from_file(filename)

        assert config_from_dict == config_from_file


@pytest.mark.integration_test
@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@pytest.mark.usefixtures("set_site_config")
@settings(max_examples=20)
@given(config_generators())
def test_that_ert_config_is_serializable(tmp_path_factory, config_generator):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_values:
        ert_config = ErtConfig.from_dict(
            config_values.to_config_dict("config.ert", os.getcwd())
        )
        config_json = json.loads(RootModel[ErtConfig](ert_config).model_dump_json())
        from_json = ErtConfig(**config_json)
        assert from_json == ert_config


def test_that_ert_config_has_valid_schema():
    TypeAdapter(ErtConfig).json_schema()


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@pytest.mark.usefixtures("set_site_config")
@pytest.mark.integration_test
@settings(max_examples=10)
@given(config_generators())
def test_that_parsing_ert_config_result_in_expected_values(
    tmp_path_factory, config_generator
):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_values:
        ert_config = ErtConfig.from_file(filename)
        assert ert_config.ens_path == config_values.enspath
        assert ert_config.random_seed == config_values.random_seed
        assert ert_config.queue_config.max_submit == config_values.max_submit

        # If there is a job_script in queue_option it has precedence
        job_script = config_values.job_script
        for e in config_values.queue_option:
            if ert_config.queue_config.queue_system in e and "JOB_SCRIPT" in e:
                job_script = e[e.index("JOB_SCRIPT") + 1]

        assert ert_config.queue_config.queue_options.job_script == job_script
        assert ert_config.user_config_file == os.path.abspath(filename)
        assert ert_config.config_path == os.getcwd()
        assert str(ert_config.runpath_file) == os.path.abspath(
            config_values.runpath_file
        )
        assert (
            ert_config.runpath_config.num_realizations == config_values.num_realizations
        )


def test_default_ens_path():
    # By default, the ensemble path is set to 'storage'
    default_ens_path = ErtConfig.from_file_contents("NUM_REALIZATIONS 1\n").ens_path
    # Set the ENSPATH in the config file
    set_in_file_ens_path = ErtConfig.from_file_contents(
        "NUM_REALIZATIONS 1\nENSPATH storage\n"
    ).ens_path

    assert os.path.abspath(default_ens_path) == os.path.abspath(set_in_file_ens_path)

    dict_set_ens_path = ErtConfig.from_dict(
        {
            ConfigKeys.NUM_REALIZATIONS: 1,
            "ENSPATH": os.path.join(os.getcwd(), "storage"),
        }
    ).ens_path

    assert os.path.abspath(dict_set_ens_path) == os.path.abspath(default_ens_path)


@pytest.mark.parametrize(
    "max_running_value, expected_error",
    [
        (100, None),
        (-1, "Input should be greater than or equal to 0"),
        ("not_an_integer", "Input should be a valid integer"),
    ],
)
def test_queue_config_max_running_invalid_values(max_running_value, expected_error):
    contents = dedent(
        f"""
        NUM_REALIZATIONS  1
        DEFINE <STORAGE> storage/<CONFIG_FILE_BASE>-<DATE>
        RUNPATH <STORAGE>/runpath/realization-<IENS>/iter-<ITER>
        ENSPATH <STORAGE>/ensemble
        QUEUE_SYSTEM LOCAL
        QUEUE_OPTION LOCAL MAX_RUNNING {max_running_value}
        """
    )
    if expected_error:
        with pytest.raises(
            expected_exception=ConfigValidationError,
            match=expected_error,
        ):
            ErtConfig.from_file_contents(contents)
    else:
        ErtConfig.from_file_contents(contents)


def test_negative_std_cutoff_raises_validation_error():
    with pytest.raises(
        ConfigValidationError, match="'STD_CUTOFF' must have a positive float"
    ):
        ErtConfig.from_file_contents(
            dedent(
                """
                NUM_REALIZATIONS  1
                STD_CUTOFF -1
                """
            )
        )


def test_that_non_existent_job_directory_gives_config_validation_error():
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="Unable to locate job directory",
    ):
        ErtConfig.from_file_contents(
            dedent(
                """
                NUM_REALIZATIONS  1
                INSTALL_JOB_DIRECTORY does_not_exist
                """
            )
        )


def test_that_empty_job_directory_gives_warning(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.warns(ConfigWarning, match="No files found in job directory"):
        _ = ErtConfig.from_file_contents(
            dedent(
                f"""
                NUM_REALIZATIONS  1
                INSTALL_JOB_DIRECTORY {tmp_path / "empty"}
                """
            )
        )


def test_that_recursive_define_warns_when_early_liveness_detection_mechanism_triggers():
    with pytest.warns(
        ConfigWarning,
        match="Gave up replacing in runpath/realization-<IENS>/iter-<ITER>/<A>."
        "\nAfter replacing the value is now: "
        "runpath/realization-<IENS>/iter-<ITER>/<A>.\n"
        "This still contains the replacement value: <A>, "
        "which would be replaced by <A>. Probably this causes a loop.",
    ):
        _ = ErtConfig.from_file_contents(
            dedent(
                """
                NUM_REALIZATIONS  1
                DEFINE <A> <A>
                RUNPATH runpath/realization-<IENS>/iter-<ITER>/<A>
                """
            )
        )


def test_that_loading_non_existent_workflow_gives_validation_error():
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match='Cannot find file or directory "does_not_exist"',
    ):
        ErtConfig.from_file_contents(
            dedent(
                """
                NUM_REALIZATIONS  1
                LOAD_WORKFLOW does_not_exist
                """
            )
        )


def test_that_loading_non_existent_workflow_job_gives_validation_error():
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match='Cannot find file or directory "does_not_exist"',
    ):
        ErtConfig.from_file_contents(
            dedent(
                """
                NUM_REALIZATIONS  1
                LOAD_WORKFLOW_JOB does_not_exist
                """
            )
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_job_definition_file_with_unexecutable_script_gives_validation_error():
    test_config_file_name = "test.ert"
    job_script_file = os.path.abspath("not_executable")
    job_name = "JOB_NAME"
    with open(job_name, "w", encoding="utf-8") as fh:
        fh.write(f"EXECUTABLE {job_script_file}\n")
    with open(job_script_file, "w", encoding="utf-8") as fh:
        fh.write("#!/bin/sh\n")
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(
            dedent(
                f"""
                NUM_REALIZATIONS  1
                LOAD_WORKFLOW_JOB {job_name}
                """
            )
        )
    with pytest.raises(
        expected_exception=ConfigValidationError,
    ):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.parametrize("c", ["\\", "?", "+", ":", "*"])
@pytest.mark.filterwarnings("ignore:RUNPATH keyword contains no value placeholders")
@pytest.mark.usefixtures("use_tmpdir")
def test_char_in_unquoted_is_allowed(c):
    ert_config = ErtConfig.from_file_contents(
        dedent(
            f"""
            NUM_REALIZATIONS 1
            RUNPATH path{c}a/b
            """
        )
    )
    assert f"path{c}a/b" in ert_config.runpath_config.runpath_format_string


@pytest.mark.usefixtures("use_tmpdir")
def test_that_magic_strings_get_substituted_in_workflow():
    script_file_contents = dedent(
        """
        SCRIPT script.py
        ARGLIST <A>
        ARG_TYPE 0 INT
        """
    )
    workflow_file_contents = dedent(
        """
        script <ZERO>
        """
    )
    script_file_path = os.path.join(os.getcwd(), "script")
    workflow_file_path = os.path.join(os.getcwd(), "workflow")
    with open(script_file_path, mode="w", encoding="utf-8") as fh:
        fh.write(script_file_contents)
    with open(workflow_file_path, mode="w", encoding="utf-8") as fh:
        fh.write(workflow_file_contents)

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
    with open("config.ert", mode="w", encoding="utf-8") as fh:
        fh.write(
            dedent(
                f"""
                NUM_REALIZATIONS 1
                DEFINE <ZERO> 0
                LOAD_WORKFLOW_JOB {script_file_path} script
                LOAD_WORKFLOW {workflow_file_path}
                """
            )
        )

    ert_config = ErtConfig.from_file("config.ert")

    assert ert_config.workflows["workflow"].cmd_list[0][1] == ["0"]


def test_that_unknown_job_gives_config_validation_error():
    with pytest.raises(
        ConfigValidationError,
        match="Could not find forward model step 'NO_SUCH_FORWARD_MODEL_STEP'",
    ):
        _ = ErtConfig.from_file_contents(
            dedent(
                """
                NUM_REALIZATIONS  1
                FORWARD_MODEL NO_SUCH_FORWARD_MODEL_STEP
                """
            )
        )


def test_that_unknown_hooked_job_gives_config_validation_error():
    with pytest.raises(
        ConfigValidationError,
        match="Cannot setup hook for non-existing job name 'NO_SUCH_JOB'",
    ):
        _ = ErtConfig.from_file_contents(
            dedent(
                """
                NUM_REALIZATIONS  1
                HOOK_WORKFLOW NO_SUCH_JOB PRE_SIMULATION
                """
            )
        )


@pytest.mark.integration_test
@pytest.mark.usefixtures("set_site_config")
@settings(max_examples=10)
@given(config_generators())
def test_that_if_field_is_given_and_grid_is_missing_you_get_error(
    tmp_path_factory, config_generator
):
    with config_generator(tmp_path_factory) as config_values:
        config_dict = config_values.to_config_dict("test.ert", os.getcwd())
        del config_dict[ConfigKeys.GRID]
        assume(len(config_dict.get(ConfigKeys.FIELD, [])) > 0)
        with pytest.raises(
            ConfigValidationError,
            match="In order to use the FIELD keyword, a GRID must be supplied",
        ):
            _ = ErtConfig.from_dict(config_dict)


def test_that_include_statements_with_multiple_values_raises_error():
    with pytest.raises(
        ConfigValidationError, match="Keyword:INCLUDE must have exactly one argument"
    ):
        _ = ErtConfig.from_file_contents(
            dedent(
                """
                NUM_REALIZATIONS  1
                INCLUDE this and that and some-other
                """
            )
        )


@pytest.mark.integration_test
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.usefixtures("use_tmpdir")
@given(anystring=st.text())
def test_fm_step_config_via_plugin_ends_up_json_data(monkeypatch, anystring):
    assume(all(bracket not in anystring for bracket in "<>"))
    monkeypatch.setattr(
        ErtPluginManager,
        "get_forward_model_configuration",
        MagicMock(return_value={"SOME_STEP": {"FOO": anystring}}),
    )
    ert_config = ErtConfig.with_plugins().from_dict(
        {
            "INSTALL_JOB": [["SOME_STEP", ("SOME_STEP", "EXECUTABLE fm_dispatch.py")]],
            "FORWARD_MODEL": [["SOME_STEP"]],
        }
    )
    step_json = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        run_id=None,
    )
    assert step_json["jobList"][0]["environment"]["FOO"] == anystring


@pytest.mark.usefixtures("use_tmpdir")
def test_fm_step_config_via_plugin_does_not_leak_to_other_step(monkeypatch):
    monkeypatch.setattr(
        ErtPluginManager,
        "get_forward_model_configuration",
        MagicMock(return_value={"SOME_STEP": {"FOO": "bar"}}),
    )
    ert_config = ErtConfig.with_plugins().from_dict(
        {
            "INSTALL_JOB": [
                ["SOME_OTHER_STEP", ("SOME_OTHER_STEP", "EXECUTABLE fm_dispatch.py")]
            ],
            "FORWARD_MODEL": [["SOME_OTHER_STEP"]],
        }
    )
    step_json = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        run_id=None,
    )

    assert "FOO" not in step_json["jobList"][0]["environment"]


@pytest.mark.usefixtures("use_tmpdir")
def test_fm_step_config_via_plugin_has_key_names_uppercased(monkeypatch):
    monkeypatch.setattr(
        ErtPluginManager,
        "get_forward_model_configuration",
        MagicMock(return_value={"SOME_STEP": {"foo": "bar"}}),
    )
    ert_config = ErtConfig.with_plugins().from_dict(
        {
            "INSTALL_JOB": [["SOME_STEP", ("SOME_STEP", "EXECUTABLE fm_dispatch.py")]],
            "FORWARD_MODEL": [["SOME_STEP"]],
        }
    )
    step_json = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        run_id=None,
    )

    assert step_json["jobList"][0]["environment"]["FOO"] == "bar"


@pytest.mark.usefixtures("use_tmpdir")
def test_fm_step_config_via_plugin_stringifies_python_objects(monkeypatch):
    monkeypatch.setattr(
        ErtPluginManager,
        "get_forward_model_configuration",
        MagicMock(return_value={"SOME_STEP": {"FOO": {"a_dict_as_value": 1}}}),
    )
    ert_config = ErtConfig.with_plugins().from_dict(
        {
            "INSTALL_JOB": [["SOME_STEP", ("SOME_STEP", "EXECUTABLE fm_dispatch.py")]],
            "FORWARD_MODEL": [["SOME_STEP"]],
        }
    )
    step_json = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        run_id=None,
    )
    assert step_json["jobList"][0]["environment"]["FOO"] == "{'a_dict_as_value': 1}"


@pytest.mark.usefixtures("use_tmpdir")
def test_fm_step_config_via_plugin_is_overridden_by_setenv(monkeypatch):
    monkeypatch.setattr(
        ErtPluginManager,
        "get_forward_model_configuration",
        MagicMock(
            return_value={
                "SOME_STEP": {
                    "FOO": "bar_from_plugin",
                    "STEP_LOCAL_VAR": "com_from_plugin",
                }
            }
        ),
    )
    ert_config = ErtConfig.with_plugins().from_dict(
        {
            "INSTALL_JOB": [["SOME_STEP", ("SOME_STEP", "EXECUTABLE fm_dispatch.py")]],
            "SETENV": [["FOO", "bar_from_setenv"]],
            "FORWARD_MODEL": [["SOME_STEP"]],
        }
    )
    step_json = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        run_id=None,
    )
    assert step_json["global_environment"]["FOO"] == "bar_from_setenv"
    assert "FOO" not in step_json["jobList"][0]["environment"]
    assert step_json["jobList"][0]["environment"]["STEP_LOCAL_VAR"] == "com_from_plugin"


@pytest.mark.usefixtures("use_tmpdir")
def test_setenv_will_be_substituted_in_jobs_json(monkeypatch):
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            SETENV FOO <NUM_CPU>
            NUM_CPU 2
            """
        ),
        encoding="utf-8",
    )
    ert_config = ErtConfig.with_plugins().from_file("config.ert")
    step_json = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        run_id=None,
    )
    assert step_json["global_environment"]["FOO"] == "2"


@pytest.mark.usefixtures("use_tmpdir")
def test_fm_step_config_via_plugin_does_not_override_default_env(monkeypatch):
    monkeypatch.setattr(
        ErtPluginManager,
        "get_forward_model_configuration",
        MagicMock(return_value={"SOME_STEP": {"_ERT_RUNPATH": "0"}}),
    )
    Path("SOME_STEP").write_text("EXECUTABLE /bin/ls", encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            INSTALL_JOB SOME_STEP SOME_STEP
            FORWARD_MODEL SOME_STEP()
            """
        ),
        encoding="utf-8",
    )
    ert_config = ErtConfig.with_plugins().from_file("config.ert")
    step_json = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        run_id=None,
    )
    assert (
        step_json["jobList"][0]["environment"]["_ERT_RUNPATH"]
        == "simulations/realization-0/iter-0"
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_fm_step_config_via_plugin_is_substituted_for_defines(monkeypatch):
    monkeypatch.setattr(
        ErtPluginManager,
        "get_forward_model_configuration",
        MagicMock(return_value={"SOME_STEP": {"FOO": "<SOME_ERT_DEFINE>"}}),
    )
    Path("SOME_STEP").write_text("EXECUTABLE /bin/ls", encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            DEFINE <SOME_ERT_DEFINE> define_works
            NUM_REALIZATIONS 1
            INSTALL_JOB SOME_STEP SOME_STEP
            FORWARD_MODEL SOME_STEP()
            """
        ),
        encoding="utf-8",
    )
    ert_config = ErtConfig.with_plugins().from_file("config.ert")
    step_json = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        run_id=None,
    )
    assert step_json["jobList"][0]["environment"]["FOO"] == "define_works"


@pytest.mark.usefixtures("use_tmpdir")
def test_fm_step_config_via_plugin_is_dropped_if_not_define_exists(monkeypatch):
    monkeypatch.setattr(
        ErtPluginManager,
        "get_forward_model_configuration",
        MagicMock(return_value={"SOME_STEP": {"FOO": "<WILL_NOT_BE_DEFINED>"}}),
    )
    Path("SOME_STEP").write_text("EXECUTABLE /bin/ls", encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            INSTALL_JOB SOME_STEP SOME_STEP
            FORWARD_MODEL SOME_STEP()
            """
        ),
        encoding="utf-8",
    )
    ert_config = ErtConfig.with_plugins().from_file("config.ert")
    step_json = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        run_id=None,
    )
    assert "FOO" not in step_json["jobList"][0]["environment"]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_workflows_with_errors_are_not_loaded():
    """
    The user may install several workflows with LOAD_WORKFLOW_DIRECTORY that
    does not work with the current versions of plugins installed in the system,
    but could have worked with an older or newer version of the packages installed.

    Therefore the user should be warned about workflows that have issues, and not be
    able to run those later. If a workflow with errors are hooked, then the user will
    get an error indicating that there is no such workflow.
    """
    test_config_file_name = "test.ert"
    Path("WFJOB").write_text("EXECUTABLE echo\n", encoding="utf-8")
    # intentionally misspelled WFJOB as WFJAB
    Path("wf").write_text("WFJAB hello world\n", encoding="utf-8")
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        JOBNAME JOOOOOB
        LOAD_WORKFLOW_JOB WFJOB
        LOAD_WORKFLOW wf
        """
    )

    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.warns(
        ConfigWarning,
        match=r"Encountered the following error\(s\) while reading workflow 'wf'."
        " It will not be loaded: .*WFJAB is not recognized",
    ):
        ert_config = ErtConfig.from_file(test_config_file_name)
        assert "wf" not in ert_config.workflows


@pytest.mark.usefixtures("use_tmpdir")
def test_that_workflow_job_subdirectories_are_not_loaded():
    test_config_file = Path("test.ert")
    wfjob_dir = Path("WFJOBS")
    wfjob_dir.mkdir()
    (wfjob_dir / "wfjob").write_text("EXECUTABLE echo\n", encoding="utf-8")
    (wfjob_dir / "subdir").mkdir()
    test_config_file.write_text(
        dedent(
            f"""
            NUM_REALIZATIONS  1
            JOBNAME JOOOOOB
            WORKFLOW_JOB_DIRECTORY {wfjob_dir}
            """
        ),
        encoding="utf-8",
    )

    ert_config = ErtConfig.from_file(str(test_config_file))
    assert "wfjob" in ert_config.workflow_jobs


@pytest.mark.usefixtures("use_tmpdir")
def test_that_adding_a_workflow_twice_warns():
    test_config_file_name = "test.ert"
    Path("WFJOB").write_text("EXECUTABLE echo\n", encoding="utf-8")
    Path("wf").write_text("WFJOB hello world\n", encoding="utf-8")
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        JOBNAME JOOOOOB
        LOAD_WORKFLOW_JOB WFJOB
        LOAD_WORKFLOW wf
        LOAD_WORKFLOW wf
        """
    )

    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.warns(
        ConfigWarning,
        match=r"Workflow 'wf' was added twice",
    ) as warnlog:
        _ = ErtConfig.from_file(test_config_file_name)

    assert any("test.ert: Line 6" in str(w.message) for w in warnlog)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "load_statement", ["LOAD_WORKFLOW_JOB wfs/WFJOB", "WORKFLOW_JOB_DIRECTORY wfs"]
)
def test_that_failing_to_load_ert_script_with_errors_fails_gracefully(load_statement):
    """
    The user may install several workflow jobs with LOAD_WORKFLOW_JOB_DIRECTORY that
    does not work with the current versions of plugins installed in the system,
    but could have worked with an older or newer version of the packages installed.

    Therefore the user should be warned about workflow jobs that have issues, and not be
    able to run those later.
    """
    test_config_file_name = "test.ert"
    Path("wfs").mkdir()
    Path("wfs/WFJOB").write_text("SCRIPT wf_script.py\nINTERNAL True", encoding="utf-8")
    Path("wf_script.py").write_text("", encoding="utf-8")
    test_config_contents = dedent(
        f"""
        NUM_REALIZATIONS  1
        {load_statement}
        """
    )

    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with (
        pytest.warns(
            ConfigWarning, match="Loading workflow job.*failed.*It will not be loaded."
        ),
        pytest.warns(ConfigWarning, match="Deprecated keywords, SCRIPT and INTERNAL"),
    ):
        ert_config = ErtConfig.from_file(test_config_file_name)
        assert "wf" not in ert_config.workflows


def test_that_define_statements_with_less_than_one_argument_raises_error():
    with pytest.raises(
        ConfigValidationError,
        match=r"DEFINE must have (two or more|at least 2) arguments",
    ):
        _ = ErtConfig.from_file_contents(
            dedent(
                """
                NUM_REALIZATIONS  1
                DEFINE <USER>
                """
            )
        )


def test_that_define_statements_with_more_than_one_argument():
    ert_config = ErtConfig.from_file_contents(
        dedent(
            """
            NUM_REALIZATIONS  1
            DEFINE <TEST1> 111 222 333
            DEFINE <TEST2> <TEST1> 444 555
            """
        )
    )
    assert ert_config.substitutions.get("<TEST1>") == "111 222 333"
    assert ert_config.substitutions.get("<TEST2>") == "111 222 333 444 555"


def test_that_define_statements_works_with_empty_string():
    ert_config = ErtConfig.from_file_contents(
        dedent(
            """
            NUM_REALIZATIONS  1
            DEFINE <TEST1> ""
            """
        )
    )
    assert ert_config.substitutions.get("<TEST1>") == ""  # noqa


@pytest.mark.usefixtures("use_tmpdir")
def test_that_include_statements_work():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        INCLUDE include.ert
        """
    )
    test_include_file_name = "include.ert"
    test_include_contents = dedent(
        """
        JOBNAME included
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with open(test_include_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_include_contents)

    ert_config = ErtConfig.from_file(test_config_file_name)
    assert ert_config.runpath_config.jobname_format_string == "included"


@pytest.mark.usefixtures("use_tmpdir")
def test_that_define_string_quotes_are_removed():
    ert_Config = ErtConfig.from_file_contents(
        dedent(
            """
            DEFINE <A> "A"
            NUM_REALIZATIONS 1
            """
        )
    )
    assert ert_Config.substitutions.get("<A>") == "A"


@pytest.mark.usefixtures("use_tmpdir")
def test_that_included_files_uses_paths_relative_to_itself():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS 1
        INCLUDE includes/install_jobs.ert
        """
    )
    os.mkdir("includes")
    test_include_file_name = "includes/install_jobs.ert"
    test_include_contents = dedent(
        """
        INSTALL_JOB FM ../FM
        """
    )
    test_fm_file_name = "FM"
    test_fm_contents = dedent(
        """
        EXECUTABLE echo
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with open(test_include_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_include_contents)
    with open(test_fm_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_fm_contents)

    ert_config = ErtConfig.from_file(test_config_file_name)
    assert ert_config.installed_forward_model_steps["FM"].name == "FM"


@pytest.mark.parametrize("val, expected", [("TrUe", True), ("FaLsE", False)])
def test_that_boolean_values_can_be_any_case(val, expected):
    ert_config = ErtConfig.from_file_contents(
        dedent(
            f"""
            NUM_REALIZATIONS  1
            STOP_LONG_RUNNING {val}
            """
        )
    )
    assert ert_config.queue_config.stop_long_running == expected


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
def test_that_include_take_into_account_path():
    """
    Tests that use_new_parser resolves an issue
    with the old parser where the first relative path
    FORWARD_MODEL is chosen for all.
    """
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        INCLUDE dir/include.ert
        INSTALL_JOB job2 job2
        """
    )
    test_include_file_name = "dir/include.ert"
    test_include_contents = dedent(
        """
        INSTALL_JOB job1 job1
        """
    )
    # The old parser tries to find dir/job2
    os.mkdir("dir")
    Path("dir/job1").write_text("EXECUTABLE echo\n", encoding="utf-8")
    Path("job2").write_text("EXECUTABLE ls\n", encoding="utf-8")
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with open(test_include_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_include_contents)

    ert_config = ErtConfig.from_file(test_config_file_name)
    assert list(ert_config.installed_forward_model_steps.keys()) == [
        "job1",
        "job2",
    ]


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
def test_that_substitution_happens_for_include():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        DEFINE <file> include.ert
        INCLUDE dir/<file>
        """
    )
    test_include_file_name = "dir/include.ert"
    test_include_contents = dedent(
        """
        RUNPATH my_silly_runpath<ITER>-<IENS>
        """
    )
    os.mkdir("dir")
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with open(test_include_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_include_contents)

    ert_config = ErtConfig.from_file(test_config_file_name)
    assert (
        "my_silly_runpath<ITER>-<IENS>"
        in ert_config.runpath_config.runpath_format_string
    )


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
def test_that_defines_in_included_files_has_immediate_effect():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        INCLUDE dir/include.ert
        RUNPATH <FOO>-<ITER>-<IENS>
        DEFINE <FOO> bar
        """
    )
    test_include_file_name = "dir/include.ert"
    test_include_contents = dedent(
        """
        DEFINE <FOO> baz
        """
    )
    os.mkdir("dir")
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with open(test_include_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_include_contents)

    ert_config = ErtConfig.from_file(test_config_file_name)
    assert "baz-<ITER>-<IENS>" in ert_config.runpath_config.runpath_format_string


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_multiple_errors_are_shown_for_forward_model():
    with pytest.raises(ConfigValidationError) as err:
        _ = ErtConfig.from_file_contents(
            dedent(
                """
                NUM_REALIZATIONS  1
                FORWARD_MODEL does_not_exist
                FORWARD_MODEL does_not_exist2
                """
            )
        )

    expected_nice_messages_list = [
        (
            "Line 3 (Column 15-29): "
            "Could not find forward model step 'does_not_exist' "
            "in list of installed forward model steps: []"
        ),
        (
            "Line 4 (Column 15-30): "
            "Could not find forward model step 'does_not_exist2' "
            "in list of installed forward model steps: []"
        ),
    ]

    cli_message = err.value.cli_message()

    for msg in expected_nice_messages_list:
        assert any(line.endswith(msg) for line in cli_message.splitlines())


@pytest.mark.usefixtures("use_tmpdir")
def test_that_redefines_work_with_setenv():
    ert_config = ErtConfig.from_file_contents(
        dedent(
            """
            NUM_REALIZATIONS 1
            DEFINE <X> 3
            SETENV VAR <X>
            DEFINE <X> 4
            SETENV VAR2 <X>
            """
        )
    )

    assert ert_config.env_vars["VAR"] == "3"
    assert ert_config.env_vars["VAR2"] == "4"


@pytest.mark.usefixtures("use_tmpdir")
def test_parsing_workflow_with_multiple_args():
    script_file_contents = dedent(
        """
        SCRIPT script.py
        ARGLIST <A> -r <B> -t <C>
        """
    )
    workflow_file_contents = dedent(
        """
        script <ZERO>
        """
    )
    script_file_path = os.path.join(os.getcwd(), "script")
    workflow_file_path = os.path.join(os.getcwd(), "workflow")
    with open(script_file_path, mode="w", encoding="utf-8") as fh:
        fh.write(script_file_contents)
    with open(workflow_file_path, mode="w", encoding="utf-8") as fh:
        fh.write(workflow_file_contents)

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
    with open("config.ert", mode="w", encoding="utf-8") as fh:
        fh.write(
            dedent(
                f"""
                NUM_REALIZATIONS 1
                DEFINE <ZERO> 0
                LOAD_WORKFLOW_JOB {script_file_path} script
                LOAD_WORKFLOW {workflow_file_path}
                """
            )
        )

    ert_config = ErtConfig.from_file("config.ert")

    assert ert_config is not None


@pytest.mark.parametrize("parameter", ["<ECLBASE>", "<RUNPATH>"])
def test_no_warning_given_when_using_parameters_defined_by_ert_in_forward_model_steps(
    caplog, recwarn, parameter
):
    """
    This is a regression test for a bug where users would get a warning when
    they used the parameters <ECLBASE> or <RUNPATH> as these were defined
    after the arguments to the forward model step was checked.

    """
    caplog.set_level(logging.WARNING)
    ErtConfig.from_dict(
        {
            "INSTALL_JOB": [
                ["name", ("file", f"EXECUTABLE echo\nARGLIST {parameter}")]
            ],
            "FORWARD_MODEL": [["name", [(f"{parameter}=", f"A/{parameter}")]]],
        }
    )
    for w in recwarn:
        assert not issubclass(w.category, ConfigWarning)


@pytest.mark.usefixtures("use_tmpdir")
def test_validate_no_logs_when_overwriting_with_same_value(caplog):
    with open("step_file", "w", encoding="utf-8") as fout:
        fout.write("EXECUTABLE echo\nARGLIST <VAR1> <VAR2> <VAR3>\n")

    with caplog.at_level(logging.INFO):
        ert_config = ErtConfig.from_file_contents(
            """
            NUM_REALIZATIONS 1
            DEFINE <VAR1> 10
            DEFINE <VAR2> 20
            DEFINE <VAR3> 55
            INSTALL_JOB step_name step_file
            FORWARD_MODEL step_name(<VAR1>=10, <VAR2>=<VAR2>, <VAR3>=5)
            """
        )
        create_forward_model_json(
            context=ert_config.substitutions,
            forward_model_steps=ert_config.forward_model_steps,
            env_vars=ert_config.env_vars,
            user_config_file=ert_config.user_config_file,
            run_id="0",
            iens=0,
            itr=0,
        )

    assert (
        (
            "Private arg '<VAR3>':'5' chosen over global '55' "
            "in forward model step step_name"
        )
        in caplog.text
        and (
            "Private arg '<VAR1>':'10' chosen over global '10' "
            "in forward model step step_name"
        )
        not in caplog.text
        and (
            "Private arg '<VAR2>':'20' chosen over global '20' "
            "in forward model step step_name"
        )
        not in caplog.text
    )


@pytest.mark.parametrize(
    "obsolete_analysis_keyword,error_msg",
    [
        ("USE_EE", "Keyword USE_EE has been replaced"),
        ("USE_GE", "Keyword USE_GE has been replaced"),
        ("ENKF_NCOMP", r"ENKF_NCOMP keyword\(s\) has been removed"),
        (
            "ENKF_SUBSPACE_DIMENSION",
            r"ENKF_SUBSPACE_DIMENSION keyword\(s\) has been removed",
        ),
    ],
)
def test_that_removed_analysis_module_keywords_raises_error(
    obsolete_analysis_keyword, error_msg
):
    with pytest.raises(
        ConfigValidationError,
        match=error_msg,
    ):
        _ = ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\n"
            f"ANALYSIS_SET_VAR STD_ENKF {obsolete_analysis_keyword} 1\n"
        )


def test_ert_config_parser_fails_gracefully_on_unreadable_config_file(caplog, tmp_path):
    config_file = Path(tmp_path) / "config.ert"
    config_file.write_text("")
    os.chmod(config_file, 0o000)
    caplog.set_level(logging.WARNING)

    with pytest.raises(ConfigValidationError, match="Permission"):
        ErtConfig.from_file(config_file)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_context_types_are_json_serializable():
    bf = ContextBool(val=False, token=None, keyword_token=None)
    bt = ContextBool(val=True, token=None, keyword_token=None)
    i = ContextInt(val=23, token=None, keyword_token=None)
    s = ContextString(val="forty_two", token=None, keyword_token=None)
    fl = ContextFloat(val=4.2, token=None, keyword_token=None)
    cl = ContextList.with_values(None, values=[bf, bt, i, s, fl])

    payload = {
        "context_bool_false": bf,
        "context_bool_true": bt,
        "context_int": i,
        "context_str": s,
        "context_float": fl,
        "context_list": cl,
    }

    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, cls=ContextBoolEncoder)

    with open("test.json", encoding="utf-8") as f:
        r = json.load(f)

    assert isinstance(r["context_bool_false"], bool)
    assert isinstance(r["context_bool_true"], bool)
    assert r["context_bool_false"] is False
    assert r["context_bool_true"] is True
    assert isinstance(r["context_int"], int)
    assert isinstance(r["context_str"], str)
    assert isinstance(r["context_float"], float)
    assert isinstance(r["context_list"], list)


def test_no_timemap_or_refcase_provides_clear_error():
    with pytest.raises(
        ConfigValidationError,
        match="Missing REFCASE or TIME_MAP for observations: GDO",
    ):
        ErtConfig.from_dict(
            {
                "GEN_DATA": [["GD", {"RESULT_FILE": "%d", "REPORT_STEPS": "1"}]],
                "OBS_CONFIG": (
                    "obs_config",
                    """
                    GENERAL_OBSERVATION GDO {
                       DATA       = GD;
                       INDEX_LIST = 0;
                       DATE       = 2015-06-13;
                       VALUE      = 0.0;
                       ERROR      = 0.1;
                    };
                    """,
                ),
            }
        )


def test_that_multiple_errors_are_shown_when_validating_observation_config():
    with pytest.raises(ConfigValidationError) as err:
        ErtConfig.from_dict(
            {
                "GEN_DATA": [["GD", {"RESULT_FILE": "%d", "REPORT_STEPS": "1"}]],
                "OBS_CONFIG": (
                    "obs_config",
                    """
                    SUMMARY_OBSERVATION SUM1
                    {
                        VALUE   = 0.7;
                        ERROR   = 0.07;
                        DATE    = 2010-12-26;
                    };

                    SUMMARY_OBSERVATION SUM2
                    {
                        ERROR   = 0.05;
                        DATE    = 2011-12-21;
                        KEY     = WOPR:OP1;
                    };
                    """,
                ),
            }
        )
    expected_errors = [
        'Line 2 (Column 41-45): Missing item "KEY" in SUM1',
        'Line 9 (Column 41-45): Missing item "VALUE" in SUM2',
    ]

    for error in expected_errors:
        assert error in str(err.value)


@pytest.mark.usefixtures("copy_snake_oil_case")
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_multiple_errors_are_shown_when_generating_observations():
    injected_errors = {
        7: "    RESTART = 0;",
        15: "    DATE    = 2010-01-01;",
        31: "    DATE    = 2009-12-15;",
        39: "    DATE    = 2010-12-10;",
    }
    with fileinput.input("observations/observations.txt", inplace=True) as fin:
        for line_number, line in enumerate(fin, 1):
            if line_number in injected_errors:
                print(injected_errors[line_number])
                continue
            print(line, end="")

    with pytest.raises(ConfigValidationError) as err:
        _ = ErtConfig.from_file("snake_oil.ert")

    expected_errors = [
        "Line 3 (Column 21-31): It is unfortunately not possible",
        "Line 11 (Column 21-32): It is unfortunately not possible",
        "Line 27 (Column 21-33): Problem with date in summary observation WOPR_OP1_108",
        "Line 35 (Column 21-33): Problem with date in summary observation WOPR_OP1_144",
    ]

    for error in expected_errors:
        assert error in str(err.value)


def test_job_name_with_slash_fails_validation():
    with pytest.raises(ConfigValidationError, match="JOBNAME cannot contain '/'"):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 100\nJOBNAME dir/eclbase")


def test_using_relative_path_to_eclbase_sets_jobname_to_basename():
    assert (
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 100\nECLBASE dir/eclbase_%d"
        ).runpath_config.jobname_format_string
        == "eclbase_<IENS>"
    )


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@pytest.mark.parametrize(
    "param_config", ["coeffs_priors", "template.txt output.txt coeffs_priors"]
)
def test_that_empty_params_file_gives_reasonable_error(tmpdir, param_config):
    with tmpdir.as_cwd():
        config = """
        NUM_REALIZATIONS 1
        GEN_KW COEFFS """
        config += param_config

        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)

        # Create an empty file named 'coeffs_priors'
        with open("coeffs_priors", mode="w", encoding="utf-8") as fh:
            pass

        # Create an empty file named 'template.txt'
        with open("template.txt", mode="w", encoding="utf-8") as fh:
            pass

        with pytest.raises(ConfigValidationError, match="No parameters specified in"):
            ErtConfig.from_file("config.ert")


@pytest.mark.parametrize(
    "max_running_queue_config_entry",
    [
        pytest.param(
            """
            MAX_RUNNING 6
            QUEUE_OPTION LSF MAX_RUNNING 2
            QUEUE_OPTION SLURM MAX_RUNNING 2
            """,
            id="general_keyword_max_running",
        ),
        pytest.param(
            """
            QUEUE_OPTION TORQUE MAX_RUNNING 6
            MAX_RUNNING 2
            """,
            id="queue_option_max_running",
        ),
    ],
)
def test_queue_config_max_running_queue_option_has_priority_over_general_option(
    max_running_queue_config_entry,
):
    assert (
        ErtConfig.from_file_contents(
            dedent(
                f"""
                NUM_REALIZATIONS  100
                DEFINE <STORAGE> storage/<CONFIG_FILE_BASE>-<DATE>
                RUNPATH <STORAGE>/runpath/realization-<IENS>/iter-<ITER>
                ENSPATH <STORAGE>/ensemble
                QUEUE_SYSTEM TORQUE
                {max_running_queue_config_entry}
                """
            )
        ).queue_config.max_running
        == 6
    )


def test_general_option_in_local_config_has_priority_over_site_config():
    config = ErtConfig.from_file_contents(
        user_config_contents=dedent(
            """
            NUM_REALIZATIONS  100
            DEFINE <STORAGE> storage/<CONFIG_FILE_BASE>-<DATE>
            RUNPATH <STORAGE>/runpath/realization-<IENS>/iter-<ITER>
            ENSPATH <STORAGE>/ensemble
            QUEUE_SYSTEM TORQUE
            MAX_RUNNING 13
            SUBMIT_SLEEP 14
            """
        ),
        site_config_contents=dedent(
            """
        QUEUE_OPTION TORQUE MAX_RUNNING 6
        QUEUE_SYSTEM LOCAL
        QUEUE_OPTION TORQUE SUBMIT_SLEEP 7
        """
        ),
    )
    assert config.queue_config.max_running == 13
    assert config.queue_config.submit_sleep == 14
    assert config.queue_config.queue_system == QueueSystem.TORQUE


def test_warning_raised_when_summary_key_and_no_simulation_job_present():
    with warnings.catch_warnings(record=True) as all_warnings:
        ErtConfig.from_dict(
            {
                "SUMMARY": ["*"],
                "ECLBASE": "RESULT_SUMMARY",
                "INSTALL_JOB": [["name", ("file", "EXECUTABLE echo")]],
                "FORWARD_MODEL": [["name"]],
            }
        )

    assert any(
        str(w.message)
        == (
            "Config contains a SUMMARY key but no forward model "
            "steps known to generate a summary file"
        )
        for w in all_warnings
        if isinstance(w.message, ConfigWarning)
    )


@pytest.mark.parametrize(
    "job_name",
    [
        "eclipse",
        "eclipse100",
        "flow",
        "FLOW",
        "ECLIPSE100",
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_no_warning_when_summary_key_and_simulation_job_present(job_name):
    # forward model step with <ECLBASE> and <RUNPATH> as arguments
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConfigWarning)
        ErtConfig.from_dict(
            {
                "SUMMARY": ["*"],
                "ECLBASE": "RESULT_SUMMARY",
                "INSTALL_JOB": [[job_name, ("file", "EXECUTABLE echo")]],
                "FORWARD_MODEL": [[job_name]],
            }
        )


def test_warning_is_emitted_when_malformatted_runpath():
    with warnings.catch_warnings(record=True, category=ConfigWarning) as all_warnings:
        ErtConfig.from_file_contents(
            dedent(
                """\
                NUM_REALIZATIONS 1
                RUNPATH <STORAGE>/runpath/constant-realization-num/constant-iter-num
                """
            )
        )
    assert any(
        ("RUNPATH keyword contains no value placeholders" in str(w.message))
        for w in all_warnings
    )


@given(string=st.text(), section_length=st.integers())
def test_split_string_into_sections(string, section_length):
    split_string = _split_string_into_sections(string, section_length)
    assert "".join(split_string) == string
    if section_length > 0:
        for section in split_string:
            assert len(section) <= section_length
    else:
        split_string = [string]


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "template_target",
    ["<ECLBASE>.DATA", "foo/bar/ECLIPSEDECK-<IENS>.DATA"],
)
def test_warning_is_emitted_for_run_template(template_target):
    Path("templates").mkdir()
    Path("templates/ECLDECK.DATA").touch()
    with pytest.warns(ConfigWarning, match="Use DATA_FILE instead of"):
        ErtConfig.from_file_contents(
            dedent(
                f"""\
                NUM_REALIZATIONS 1
                ECLBASE foo/bar/ECLIPSEDECK-<IENS>
                RUN_TEMPLATE templates/ECLDECK.DATA {template_target}
                """
            )
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_warning_is_emitted_for_ecl_base_in_run_template():
    Path("templates").mkdir()
    Path("templates/ECLDECK.DATA").touch()
    with pytest.warns(
        ConfigWarning, match="Substitution template <ECL_BASE> is deprecated"
    ):
        ErtConfig.from_file_contents(
            dedent(
                """\
                NUM_REALIZATIONS 1
                ECLBASE foo/bar/ECLIPSEDECK-<IENS>
                RUN_TEMPLATE templates/ECLDECK.DATA <ECL_BASE>.DATA
                """
            )
        )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize("eclbase_line", ["", "ECLBASE foo/bar/DECK"])
def test_warning_is_not_emitted_for_random_run_template(eclbase_line):
    Path("templates").mkdir()
    Path("templates/ECLDECK.DATA").touch()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ErtConfig.from_file_contents(
            dedent(
                f"""\
                NUM_REALIZATIONS 1
                {eclbase_line}
                RUN_TEMPLATE templates/ECLDECK.DATA foo/bar/OTHERDECK.DATA
                """
            )
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_warning_is_not_emitted_for_when_num_cpu_is_explicit():
    Path("templates").mkdir()
    Path("templates/ECLDECK.DATA").touch()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ErtConfig.from_file_contents(
            dedent(
                """\
                NUM_REALIZATIONS 1
                NUM_CPU 8
                ECLBASE foo/bar/ECLIPSEDECK-<IENS>
                RUN_TEMPLATE templates/ECLDECK.DATA <ECLBASE>.DATA
                """
            )
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_parsing_define_within_workflow():
    script_file_contents = dedent(
        """
        SCRIPT script.py
        ARGLIST one two
        """
    )
    workflow_file_contents = dedent(
        """
        DEFINE <FOO> workflow_foo
        DEFINE <FOO2> workflow_foo2
        script <FOO> <FOO2>
        """
    )

    workflow2_file_contents = dedent(
        """
        script <FOO2> <FOO>
        """
    )

    script_file_path = os.path.join(os.getcwd(), "script")
    workflow_file_path = os.path.join(os.getcwd(), "workflow")
    workflow2_file_path = os.path.join(os.getcwd(), "workflow2")

    with open(script_file_path, mode="w", encoding="utf-8") as fh:
        fh.write(script_file_contents)

    with open(workflow_file_path, mode="w", encoding="utf-8") as fh:
        fh.write(workflow_file_contents)

    with open(workflow2_file_path, mode="w", encoding="utf-8") as fh:
        fh.write(workflow2_file_contents)

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
    with open("config.ert", mode="w", encoding="utf-8") as fh:
        fh.write(
            dedent(
                f"""
                NUM_REALIZATIONS 1
                DEFINE <FOO> ertconfig_foo
                DEFINE <FOO2> ertconfig_foo2
                LOAD_WORKFLOW_JOB {script_file_path} script
                LOAD_WORKFLOW {workflow_file_path}
                LOAD_WORKFLOW {workflow2_file_path}
                """
            )
        )

    with pytest.warns(ConfigWarning, match="SCRIPT has no effect"):
        ert_config = ErtConfig.from_file("config.ert")

    # Expect overwritten defines within workflow scope
    wf = ert_config.workflows["workflow"]
    assert wf.cmd_list[0][1] == ["workflow_foo", "workflow_foo2"]

    # Now expect the ertconfig ones, should not be overwritten
    # by workflow 1's defines outside of its scope
    wf2 = ert_config.workflows["workflow2"]
    assert wf2.cmd_list[0][1] == ["ertconfig_foo2", "ertconfig_foo"]

    assert ert_config.substitutions["<FOO>"] == "ertconfig_foo"
    assert ert_config.substitutions["<FOO2>"] == "ertconfig_foo2"


def test_design2params_also_validates_design_matrix(tmp_path, caplog, monkeypatch):
    design_matrix_file = tmp_path / "my_design_matrix.xlsx"
    _create_design_matrix(
        design_matrix_file,
        pl.DataFrame(
            {
                "REAL": ["not_a_valid_real"],
                "a": [1],
                "category": ["cat1"],
            }
        ),
        pl.DataFrame([["b", 1], ["c", 2]], orient="row"),
    )
    mock_design2params = ForwardModelStep(
        "DESIGN2PARAMS",
        "/usr/bin/env",
        arglist=["<IENS>", "<xls_filename>", "<designsheet>", "<defaultssheet>"],
    )
    monkeypatch.setattr(
        ErtConfig,
        "PREINSTALLED_FORWARD_MODEL_STEPS",
        {"DESIGN2PARAMS": mock_design2params},
    )
    ErtConfig.from_file_contents(
        f"""\
            NUM_REALIZATIONS 1
            FORWARD_MODEL DESIGN2PARAMS(<xls_filename>={design_matrix_file}, \\
                <designsheet>=DesignSheet,<defaultssheet>=DefaultSheet)"""
    )
    assert "DESIGN_MATRIX validation of DESIGN2PARAMS" in caplog.text


def test_two_design2params_validates_design_matrix_merging(
    tmp_path, caplog, monkeypatch
):
    design_matrix_file = tmp_path / "my_design_matrix.xlsx"
    design_matrix_file2 = tmp_path / "my_design_matrix2.xlsx"
    _create_design_matrix(
        design_matrix_file,
        pl.DataFrame(
            {
                "REAL": [0, 1],
                "letters": ["x", "y"],
            }
        ),
        pl.DataFrame([["a", 1], ["c", 2]], orient="row"),
    )
    _create_design_matrix(
        design_matrix_file2,
        pl.DataFrame({"REAL": [1, 2], "numbers": [99, 98]}),
        pl.DataFrame(),
    )
    mock_design2params = ForwardModelStep(
        "DESIGN2PARAMS",
        "/usr/bin/env",
        arglist=["<IENS>", "<xls_filename>", "<designsheet>", "<defaultssheet>"],
    )
    monkeypatch.setattr(
        ErtConfig,
        "PREINSTALLED_FORWARD_MODEL_STEPS",
        {"DESIGN2PARAMS": mock_design2params},
    )
    ErtConfig.from_file_contents(
        f"""\
            NUM_REALIZATIONS 1
            FORWARD_MODEL DESIGN2PARAMS(<xls_filename>={design_matrix_file}, \\
                <designsheet>=DesignSheet,<defaultssheet>=DefaultSheet)
            FORWARD_MODEL DESIGN2PARAMS(<xls_filename>={design_matrix_file2}, \\
                <designsheet>=DesignSheet,<defaultssheet>=DefaultSheet)
            """
    )
    assert (
        "Design matrix merging would have failed due to: "
        f"Design Matrices '{design_matrix_file.name} (DesignSheet DefaultSheet)' "
        f"and '{design_matrix_file2.name} (DesignSheet DefaultSheet)' "
        "do not have the same active realizations" in caplog.text
    )


def test_three_design2params_validates_design_matrix_merging(
    tmp_path, caplog, monkeypatch
):
    design_matrix_file = tmp_path / "my_design_matrix.xlsx"
    design_matrix_file2 = tmp_path / "my_design_matrix2.xlsx"
    design_matrix_file3 = tmp_path / "my_design_matrix3.xlsx"
    _create_design_matrix(
        design_matrix_file,
        pl.DataFrame(
            {
                "REAL": [0, 1],
                "letters": ["x", "y"],
            }
        ),
        pl.DataFrame([["a", 1], ["c", 2]], orient="row"),
    )
    _create_design_matrix(
        design_matrix_file2,
        pl.DataFrame(
            {
                "REAL": [0, 1],
                "weirdly": ["x", "y"],
            }
        ),
        pl.DataFrame([["b", 1], ["d", 2]], orient="row"),
    )
    _create_design_matrix(
        design_matrix_file3,
        pl.DataFrame({"REAL": [1, 2], "numbers": [99, 98]}),
        pl.DataFrame(),
    )
    mock_design2params = ForwardModelStep(
        "DESIGN2PARAMS",
        "/usr/bin/env",
        arglist=["<IENS>", "<xls_filename>", "<designsheet>", "<defaultssheet>"],
    )
    monkeypatch.setattr(
        ErtConfig,
        "PREINSTALLED_FORWARD_MODEL_STEPS",
        {"DESIGN2PARAMS": mock_design2params},
    )
    ErtConfig.from_file_contents(
        f"""\
            NUM_REALIZATIONS 1
            FORWARD_MODEL DESIGN2PARAMS(<xls_filename>={design_matrix_file}, \\
                <designsheet>=DesignSheet,<defaultssheet>=DefaultSheet)
            FORWARD_MODEL DESIGN2PARAMS(<xls_filename>={design_matrix_file2}, \\
                <designsheet>=DesignSheet,<defaultssheet>=DefaultSheet)
            FORWARD_MODEL DESIGN2PARAMS(<xls_filename>={design_matrix_file3}, \\
                <designsheet>=DesignSheet,<defaultssheet>=DefaultSheet)
            """
    )
    assert (
        "Design matrix merging would have failed due to: Design Matrices "
        f"'{design_matrix_file.name} (DesignSheet DefaultSheet)' "
        f"and '{design_matrix_file3.name} (DesignSheet DefaultSheet)' "
        "do not have the same active realizations" in caplog.text
    )


def test_design_matrix_default_argument(tmp_path):
    design_matrix_file = tmp_path / "my_design_matrix.xlsx"
    _create_design_matrix(
        design_matrix_file,
        pl.DataFrame(
            {
                "REAL": [1],
                "a": [1],
                "category": ["cat1"],
            }
        ),
        pl.DataFrame([["b", 1], ["c", 2]], orient="row"),
    )

    config = ErtConfig.from_file_contents(
        f"NUM_REALIZATIONS 1\nDESIGN_MATRIX {design_matrix_file}"
    )
    assert config.analysis_config.design_matrix
    assert config.analysis_config.design_matrix.design_sheet == "DesignSheet"
    assert config.analysis_config.design_matrix.default_sheet is None


@pytest.mark.usefixtures("use_tmpdir")
def test_run_template_raises_configvalidationerror_with_more_than_two_arguments():
    Path("template.txt").touch()
    Path("input.txt").touch()
    with pytest.raises(
        ConfigValidationError, match="RUN_TEMPLATE must have maximum 2 arguments"
    ):
        ErtConfig.from_file_contents(
            dedent(
                """\
            NUM_REALIZATIONS 1
            RUN_TEMPLATE template.txt input.txt excess_argument
            """
            )
        )


@pytest.fixture
def setup_workflow_file():
    workflow_file_path = os.path.join(os.getcwd(), "workflow")
    with open(workflow_file_path, mode="w", encoding="utf-8") as fh:
        fh.write("TEST_SCRIPT")

    with open("config.ert", mode="w", encoding="utf-8") as fh:
        fh.write(
            dedent(
                f"""
                    NUM_REALIZATIONS 1

                    LOAD_WORKFLOW {workflow_file_path} workflow_alias
                    HOOK_WORKFLOW workflow_alias PRE_EXPERIMENT
                    """
            )
        )


@pytest.mark.usefixtures("use_tmpdir", "setup_workflow_file")
def test_ert_script_hook_pre_experiment_but_asks_for_storage():
    class SomeScript(ErtScript):
        def run(self, storage: Storage):
            pass

    wfjob = ErtScriptWorkflow(
        name="TEST_SCRIPT",
        ert_script=SomeScript,
    )
    ErtConfig.PREINSTALLED_WORKFLOWS = {"TEST_SCRIPT": wfjob}

    with pytest.raises(
        ConfigValidationError,
        match=r"Workflow job TEST_SCRIPT.*"
        r"expected fixtures.*storage.*",
    ):
        ErtConfig.from_file("config.ert")


@pytest.mark.usefixtures("use_tmpdir", "setup_workflow_file")
def test_ert_script_hook_pre_experiment_but_asks_for_ensemble():
    class SomeScript(ErtScript):
        def run(self, ensemble: LocalEnsemble):
            pass

    wfjob = ErtScriptWorkflow(
        name="TEST_SCRIPT",
        ert_script=SomeScript,
    )
    ErtConfig.PREINSTALLED_WORKFLOWS = {"TEST_SCRIPT": wfjob}

    with pytest.raises(
        ConfigValidationError,
        match=r"Workflow job TEST_SCRIPT.*"
        r"expected fixtures.*ensemble.*",
    ):
        ErtConfig.from_file("config.ert")


@pytest.mark.usefixtures("use_tmpdir", "setup_workflow_file")
def test_ert_script_hook_pre_experiment_but_asks_for_random_seed():
    class SomeScript(ErtScript):
        def run(self, random_seed: int):
            pass

    wfjob = ErtScriptWorkflow(
        name="TEST_SCRIPT",
        ert_script=SomeScript,
    )
    ErtConfig.PREINSTALLED_WORKFLOWS = {"TEST_SCRIPT": wfjob}

    ErtConfig.from_file("config.ert")


@pytest.mark.usefixtures("use_tmpdir", "setup_workflow_file")
def test_ert_script_hook_pre_experiment_essettings_fails():
    class SomeScript(ErtScript):
        def run(self, es_settings: ESSettings):
            pass

    wfjob = ErtScriptWorkflow(
        name="TEST_SCRIPT",
        ert_script=SomeScript,
    )
    ErtConfig.PREINSTALLED_WORKFLOWS = {"TEST_SCRIPT": wfjob}

    with pytest.raises(
        ConfigValidationError,
        match=(
            r".*It would work in these runtimes: "
            r"PRE_UPDATE, POST_UPDATE, PRE_FIRST_UPDATE.*"
        ),
    ):
        ErtConfig.from_file("config.ert")


@pytest.mark.usefixtures("use_tmpdir")
def test_ert_script_hook_valid_essettings_succeed():
    workflow_file_path = os.path.join(os.getcwd(), "workflow")
    with open(workflow_file_path, mode="w", encoding="utf-8") as fh:
        fh.write("TEST_SCRIPT")

    with open("config.ert", mode="w", encoding="utf-8") as fh:
        fh.write(
            dedent(
                f"""
                NUM_REALIZATIONS 1

                LOAD_WORKFLOW {workflow_file_path} workflow_alias
                HOOK_WORKFLOW workflow_alias PRE_UPDATE
                HOOK_WORKFLOW workflow_alias POST_UPDATE
                HOOK_WORKFLOW workflow_alias PRE_FIRST_UPDATE
                """
            )
        )

    class SomeScript(ErtScript):
        def run(self, es_settings: ESSettings):
            pass

    wfjob = ErtScriptWorkflow(
        name="TEST_SCRIPT",
        ert_script=SomeScript,
    )
    ErtConfig.PREINSTALLED_WORKFLOWS = {"TEST_SCRIPT": wfjob}
    ErtConfig.from_file("config.ert")


def test_queue_options_are_joined_after_option_name():
    assert (
        ErtConfig.from_file_contents(
            """
            NUM_REALIZATIONS 1
            QUEUE_SYSTEM LSF
            QUEUE_OPTION LSF LSF_RESOURCE select[bigmachine && Linux]
            """
        ).queue_config.queue_options.lsf_resource
        == "select[bigmachine && Linux]"
    )


def test_validation_error_on_gen_kw_with_design_matrix_group_name(tmp_path):
    design_matrix_file = tmp_path / "my_design_matrix.xlsx"
    _create_design_matrix(
        design_matrix_file,
        pl.DataFrame(
            {
                "REAL": [0, 1],
                "letters": ["x", "y"],
            }
        ),
        pl.DataFrame([["a", 1], ["c", 2]], orient="row"),
    )
    with open(tmp_path / "coeffs_priors", mode="w", encoding="utf-8") as fh:
        fh.write("a CONST 0")
    with pytest.raises(
        ConfigValidationError,
        match="Cannot have GEN_KW with group name DESIGN_MATRIX "
        "when using DESIGN_MATRIX keyword\\.",
    ):
        ErtConfig.from_file_contents(
            f"""\
                NUM_REALIZATIONS 1
                DESIGN_MATRIX {tmp_path}/my_design_matrix.xlsx
                GEN_KW DESIGN_MATRIX {tmp_path}/coeffs_priors
                """
        )


@pytest.mark.parametrize(
    "invalid_parameter_definition_name",
    [
        pytest.param("CWD UNIFORM 0 1", id="already_a_magic_string"),
        pytest.param("realization UNIFORM 0 1", id="realization_is_reserved"),
        pytest.param("ITER UNIFORM 0 1", id="ITER_is_reserved"),
        pytest.param("IENS UNIFORM 0 1", id="IENS_is_reserved"),
        pytest.param("a UNIFORM 0 1", id="already_defined_in_user_config"),
    ],
)
def test_validation_error_on_invalid_parameter_name(
    invalid_parameter_definition_name, tmp_path, caplog
):
    with open(tmp_path / "coeffs_priors", mode="w", encoding="utf-8") as fh:
        fh.write(invalid_parameter_definition_name)
    with pytest.raises(
        ConfigValidationError,
        match=(
            r"Found reserved parameter name\(s\): \w+. The names are already "
            "in use as magic strings or defined in the user config."
        ),
    ):
        ErtConfig.from_file_contents(
            f"""\
                DEFINE <a> 3
                NUM_REALIZATIONS 1
                GEN_KW COEFFS {tmp_path}/coeffs_priors
                """
        )


@pytest.mark.parametrize(
    "invalid_parameter_name",
    [
        pytest.param("CWD", id="already_a_magic_string"),
        pytest.param("ITER", id="ITER_is_reserved"),
        pytest.param("IENS", id="IENS_is_reserved"),
        pytest.param("RMS_SEED", id="already_defined_in_user_config"),
    ],
)
def test_validation_error_on_invalid_design_matrix_parameter_name(
    invalid_parameter_name, tmp_path
):
    design_matrix_file = tmp_path / "my_design_matrix.xlsx"
    _create_design_matrix(
        design_matrix_file,
        pl.DataFrame(
            {
                invalid_parameter_name: [0.55, 0.80],
            }
        ),
        pl.DataFrame([["a", 1], ["c", 2]], orient="row"),
    )

    with pytest.raises(
        ConfigValidationError,
        match=(
            r"Found reserved parameter name\(s\): \w+. The names are already "
            "in use as magic strings or defined in the user config."
        ),
    ):
        ErtConfig.from_file_contents(
            f"""\
                DEFINE <RMS_SEED> -20
                NUM_REALIZATIONS 1
                DESIGN_MATRIX {tmp_path}/my_design_matrix.xlsx
                """
        )


def test_that_the_runpath_keyword_sets_the_runpath_substitution():
    assert (
        ErtConfig.from_dict({"RUNPATH": "my/runpath"}).substitutions["<RUNPATH>"]
        == "my/runpath"
    )


@pytest.mark.parametrize("eclbase_substitution", ["<ECLBASE>", "<ECL_BASE>"])
def test_that_the_eclbase_keyword_sets_the_eclbase_substitution(eclbase_substitution):
    datafile = "input.data"
    assert (
        ErtConfig.from_dict(
            {
                "ECLBASE": datafile,
                "JOBNAME": "job<IENS>",
            }
        ).substitutions[eclbase_substitution]
        == datafile
    )
