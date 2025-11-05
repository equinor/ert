import logging
import os
import os.path
import shutil
import stat
from pathlib import Path
from textwrap import dedent

import pytest
from hypothesis import given, settings
from pydantic import TypeAdapter

from ert.base_model_context import use_runtime_plugins
from ert.config import ConfigValidationError, ConfigWarning, ErtConfig
from ert.config.ert_config import (
    create_forward_model_json,
    forward_model_step_from_config_contents,
)
from ert.config.forward_model_step import (
    ForwardModelStepJSON,
    ForwardModelStepPlugin,
    ForwardModelStepValidationError,
    SiteInstalledForwardModelStep,
    SiteOrUserForwardModelStep,
)
from ert.config.parsing import SchemaItemType
from ert.plugins import ErtRuntimePlugins, get_site_plugins

from .config_dict_generator import config_generators


@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model():
    name = "script.sh"
    Path(name).write_text("This is a script", encoding="utf-8")
    mode = os.stat(name).st_mode
    mode |= stat.S_IXUSR | stat.S_IXGRP
    os.chmod(name, stat.S_IMODE(mode))
    contents = """
        STDOUT null
        STDERR null
        EXECUTABLE script.sh
        """
    fm_step = forward_model_step_from_config_contents(contents, "CONFIG")
    assert fm_step.name == "CONFIG"
    assert fm_step.stdout_file is None
    assert fm_step.stderr_file is None

    assert fm_step.executable == os.path.join(os.getcwd(), "script.sh")
    assert os.access(fm_step.executable, os.X_OK)

    assert fm_step.min_arg is None

    fm_step = forward_model_step_from_config_contents(contents, "CONFIG", name="Step")
    assert fm_step.name == "Step"
    assert repr(fm_step).startswith("UserInstalledForwardModelStep(")


@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model_upgraded():
    name = "script.sh"
    Path(name).write_text("This is a script", encoding="utf-8")
    mode = os.stat(name).st_mode
    mode |= stat.S_IXUSR | stat.S_IXGRP
    os.chmod(name, stat.S_IMODE(mode))
    fm_step = forward_model_step_from_config_contents(
        """
        EXECUTABLE script.sh
        MIN_ARG 2
        MAX_ARG 7
        ARG_TYPE 0 INT
        ARG_TYPE 1 FLOAT
        ARG_TYPE 2 STRING
        ARG_TYPE 3 BOOL
        ARG_TYPE 4 RUNTIME_FILE
        ARG_TYPE 5 RUNTIME_INT
        """,
        "CONFIG",
    )
    assert fm_step.min_arg == 2
    assert fm_step.max_arg == 7
    argTypes = fm_step.arg_types
    assert argTypes == [
        SchemaItemType.INT,
        SchemaItemType.FLOAT,
        SchemaItemType.STRING,
        SchemaItemType.BOOL,
        SchemaItemType.RUNTIME_FILE,
        SchemaItemType.RUNTIME_INT,
        SchemaItemType.STRING,
    ]


def test_portable_exe_error_message():
    with (
        pytest.raises(ConfigValidationError, match="EXECUTABLE must be set"),
        pytest.warns(ConfigWarning, match='"PORTABLE_EXE" key is deprecated'),
    ):
        _ = forward_model_step_from_config_contents(
            "PORTABLE_EXE fm_dispatch.py", "CONFIG"
        )


def test_load_forward_model_missing_raises():
    with pytest.raises(ConfigValidationError, match="Could not find executable"):
        _ = forward_model_step_from_config_contents(
            "EXECUTABLE missing_script.sh", "CONFIG"
        )


@pytest.mark.filterwarnings("ignore:.*Unknown keyword 'EXECU'.*:UserWarning")
def test_load_forward_model_execu_missing_raises():
    with pytest.raises(ConfigValidationError, match="EXECUTABLE must be set"):
        _ = forward_model_step_from_config_contents(
            "EXECU missing_script.sh\n", "CONFIG"
        )


def test_load_forward_model_is_directory_raises():
    with pytest.raises(ConfigValidationError, match="directory"):
        _ = forward_model_step_from_config_contents("EXECUTABLE /tmp", "CONFIG")


def test_load_forward_model_foreign_raises():
    with pytest.raises(ConfigValidationError, match="File not executable"):
        _ = forward_model_step_from_config_contents("EXECUTABLE /etc/passwd", "CONFIG")


def test_forward_model_stdout_stderr_defaults_to_filename():
    forward_model = forward_model_step_from_config_contents(
        "EXECUTABLE fm_dispatch.py", "CONFIG"
    )

    assert forward_model.name == "CONFIG"
    assert forward_model.stdout_file == "CONFIG.stdout"
    assert forward_model.stderr_file == "CONFIG.stderr"


def test_forward_model_stdout_stderr_null_results_in_none():
    forward_model = forward_model_step_from_config_contents(
        """
        EXECUTABLE fm_dispatch.py
        STDIN null
        STDOUT null
        STDERR null
        """,
        "CONFIG",
    )

    assert forward_model.name == "CONFIG"
    assert forward_model.stdin_file is None
    assert forward_model.stdout_file is None
    assert forward_model.stderr_file is None


def test_that_arglist_is_parsed_correctly():
    forward_model = forward_model_step_from_config_contents(
        """
        EXECUTABLE fm_dispatch.py
        ARGLIST <A> B <C> <D> <E>
        """,
        "CONFIG",
    )

    assert forward_model.arglist == ["<A>", "B", "<C>", "<D>", "<E>"]


def test_that_default_env_is_set():
    forward_model = forward_model_step_from_config_contents(
        """
        EXECUTABLE fm_dispatch.py
        """,
        "CONFIG",
    )
    assert forward_model.environment == forward_model.default_env


def test_forward_model_arglist_with_weird_characters():
    forward_model = forward_model_step_from_config_contents(
        """
        STDERR    insert_nosim.stderr
        STDOUT    insert_nosim.stdout
        EXECUTABLE sed
        ARGLIST   -i s/^RUNSPEC.*/|RUNSPEC\\nNOSIM/ <ECLBASE>.DATA
        MIN_ARG 3
        MAX_ARG 3
        ARG_TYPE 0 STRING
        ARG_TYPE 0 STRING
        ARG_TYPE 0 STRING
        """,
        "CONFIG",
    )
    assert forward_model.environment == forward_model.default_env
    assert forward_model.arglist == [
        "-i",
        "s/^RUNSPEC.*/|RUNSPEC\\nNOSIM/",
        "<ECLBASE>.DATA",
    ]


@pytest.mark.integration_test
@pytest.mark.filterwarnings("ignore:MIN_REALIZATIONS")
@pytest.mark.filterwarnings(
    "ignore:Config contains a SUMMARY key but no forward model steps"
)
@settings(max_examples=10)
@given(config_generators())
def test_ert_config_throws_on_missing_forward_model_step(
    tmp_path_factory, config_generator
):
    with config_generator(tmp_path_factory) as config_values:
        config_values.install_job = []
        config_values.install_job_directory = []
        config_values.forward_model.append(
            ["this-is-not-the-job-you-are-looking-for", "<WAVE-HAND>=casually"]
        )

        with pytest.raises(
            expected_exception=ValueError, match="Could not find forward model step"
        ):
            _ = ErtConfig.from_dict(
                config_values.to_config_dict("test.ert", os.getcwd())
            )


@pytest.mark.integration_test
def test_that_substitutions_can_be_done_in_job_names(plugins_ert_config):
    """
    Regression test for a usage case involving setting ECL100 or ECL300
    that was broken by changes to forward_model substitutions.
    """

    ert_config = plugins_ert_config.from_file_contents(
        """
        NUM_REALIZATIONS  1
        DEFINE <ECL100OR300> E100
        FORWARD_MODEL ECLIPS<ECL100OR300>(\
        <VERSION>=2024.1, <NUM_CPU>=42, <OPTS>="-m")
        """
    )
    assert len(ert_config.forward_model_steps) == 1
    job = ert_config.forward_model_steps[0]
    assert job.name == "ECLIPSE100"


def test_parsing_forward_model_with_double_dash_is_possible(plugins_ert_config):
    """This is a regression test, making sure that we can put double dashes in strings.
    The use case is that a file name is utilized that contains two consecutive hyphens,
    which by the ert config parser used to be interpreted as a comment. In the new
    parser this is allowed"""

    res_config = plugins_ert_config.from_file_contents(
        """
        NUM_REALIZATIONS  1
        JOBNAME job_%d--hei
        FORWARD_MODEL COPY_FILE(<FROM>=foo,<TO>=something/hello--there.txt)
        """
    )
    assert res_config.runpath_config.jobname_format_string == "job_<IENS>--hei"
    assert (
        res_config.forward_model_steps[0].private_args["<TO>"]
        == "something/hello--there.txt"
    )


def test_parsing_forward_model_with_quotes_does_not_introduce_spaces(
    plugins_ert_config,
):
    """this is a regression test, making sure that we do not by mistake introduce
    spaces while parsing forward model lines that contain quotation marks

    the use case is that a file name is utilized that contains two consecutive hyphens,
    which by the ert config parser is interpreted as a comment - to circumvent the
    comment interpretation, quotation marks are used"""

    str_with_quotes = """smt/<foo>"/bar"/xx/"t--s.s"/yy/"z/z"/oo"""
    ert_config = plugins_ert_config.from_file_contents(
        dedent(
            f"""
            NUM_REALIZATIONS  1
            JOBNAME job_%d
            FORWARD_MODEL COPY_FILE(<FROM>=foo,<TO>={str_with_quotes})
            """
        )
    )
    assert list(ert_config.forward_model_steps[0].private_args.values()) == [
        "foo",
        "smt/<foo>/bar/xx/t--s.s/yy/z/z/oo",
    ]


def test_that_comments_are_ignored(plugins_ert_config):
    """This is a regression test, making sure that we can put double dashes in strings.
    The use case is that a file name is utilized that contains two consecutive hyphens,
    which by the ert config parser used to be interpreted as a comment. In the new
    parser this is allowed"""

    res_config = plugins_ert_config.from_file_contents(
        """
        NUM_REALIZATIONS  1
        --comment
        JOBNAME job_%d--hei --hei
        FORWARD_MODEL COPY_FILE(<FROM>=foo,<TO>=something/hello--there.txt)--foo
        """
    )
    assert res_config.runpath_config.jobname_format_string == "job_<IENS>--hei"
    assert (
        res_config.forward_model_steps[0].private_args["<TO>"]
        == "something/hello--there.txt"
    )


def test_that_quotations_in_forward_model_arglist_are_handled_correctly(
    plugins_ert_config,
):
    """This is a regression test, making sure that quoted strings behave consistently.
    They should all result in the same.
    See https://github.com/equinor/ert/issues/2766"""

    res_config = plugins_ert_config.from_file_contents(
        """
        NUM_REALIZATIONS  1
        FORWARD_MODEL COPY_FILE(<FROM>='some, thing', <TO>="some stuff", \
            <FILE>=file.txt)
        FORWARD_MODEL COPY_FILE(<FROM>='some, thing', <TO>='some stuff', \
            <FILE>=file.txt)
        FORWARD_MODEL COPY_FILE(<FROM>="some, thing", <TO>="some stuff", \
            <FILE>=file.txt)
        """
    )

    assert res_config.forward_model_steps[0].private_args["<FROM>"] == "some, thing"
    assert res_config.forward_model_steps[0].private_args["<TO>"] == "some stuff"
    assert res_config.forward_model_steps[0].private_args["<FILE>"] == "file.txt"

    assert res_config.forward_model_steps[1].private_args["<FROM>"] == "some, thing"
    assert res_config.forward_model_steps[1].private_args["<TO>"] == "some stuff"
    assert res_config.forward_model_steps[1].private_args["<FILE>"] == "file.txt"

    assert res_config.forward_model_steps[2].private_args["<FROM>"] == "some, thing"
    assert res_config.forward_model_steps[2].private_args["<TO>"] == "some stuff"
    assert res_config.forward_model_steps[2].private_args["<FILE>"] == "file.txt"


@pytest.mark.parametrize("quote_mismatched_arg", ['"A', 'A"', '"A""', '"'])
def test_unmatched_quotes_in_step_arg_gives_config_validation_error(
    quote_mismatched_arg, plugins_ert_config
):
    with (
        pytest.raises(ConfigValidationError, match="Did not expect character"),
    ):
        plugins_ert_config.from_file_contents(
            f"""
            NUM_REALIZATIONS 1
            FORWARD_MODEL COPY_FILE(<FROM>={quote_mismatched_arg})
            """
        )


def test_that_positional_forward_model_args_gives_config_validation_error():
    with pytest.raises(ConfigValidationError, match="Did not expect token: <IENS>"):
        _ = ErtConfig.from_file_contents(
            """
            NUM_REALIZATIONS  1
            FORWARD_MODEL RMS <IENS>
            """
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_installing_two_forward_model_steps_with_the_same_name_warn():
    test_config_file_name = "test.ert"
    Path("job").write_text("EXECUTABLE echo\n", encoding="utf-8")
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS 1
        INSTALL_JOB job job
        INSTALL_JOB job job
        """
    )
    Path(test_config_file_name).write_text(test_config_contents, encoding="utf-8")

    with pytest.warns(ConfigWarning, match="Duplicate forward model step"):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.fixture(scope="module")
def plugins_ert_config():
    return ErtConfig.with_plugins(get_site_plugins())


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
def test_that_forward_model_substitution_does_not_warn_about_reaching_max_iterations(
    caplog, plugins_ert_config
):
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS 1
        FORWARD_MODEL ECLIPSE100(<VERSION>=2020.2)
        """
    )
    Path(test_config_file_name).write_text(test_config_contents, encoding="utf-8")

    ert_config = plugins_ert_config.from_file(test_config_file_name)
    with caplog.at_level(logging.WARNING):
        create_forward_model_json(
            context=ert_config.substitutions,
            forward_model_steps=ert_config.forward_model_steps,
            env_vars=ert_config.env_vars,
            user_config_file=ert_config.user_config_file,
            run_id=None,
            iens=0,
            itr=0,
        )

        assert "Reached max iterations" not in caplog.text


@pytest.mark.usefixtures("use_tmpdir")
def test_that_installing_two_forward_model_steps_with_the_same_name_warn_with_dir():
    test_config_file_name = "test.ert"
    os.mkdir("jobs")
    Path("jobs/job").write_text("EXECUTABLE echo\n", encoding="utf-8")
    Path("job").write_text("EXECUTABLE echo\n", encoding="utf-8")
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS 1
        INSTALL_JOB_DIRECTORY jobs
        INSTALL_JOB job job
        """
    )
    Path(test_config_file_name).write_text(test_config_contents, encoding="utf-8")

    with pytest.warns(ConfigWarning, match="Duplicate forward model step"):
        _ = ErtConfig.from_file(test_config_file_name)


def test_that_spaces_in_forward_model_args_are_dropped(plugins_ert_config):
    # Intentionally inserted several spaces before comma
    ert_config = plugins_ert_config.from_file_contents(
        """
        NUM_REALIZATIONS  1
        FORWARD_MODEL ECLIPSE100(<VERSION>=2024.1                    , <NUM_CPU>=42)
        """
    )
    assert len(ert_config.forward_model_steps) == 1
    job = ert_config.forward_model_steps[0]
    assert job.private_args.get("<VERSION>") == "2024.1"


@pytest.mark.usefixtures("use_tmpdir")
def test_that_forward_model_with_different_token_kinds_are_added():
    """
    This is a regression tests for a problem where the parser had different
    token kinds which ended up in separate keys in the input dictionary, and were
    therefore not added
    """
    test_config_file_name = "test.ert"
    Path("job").write_text("EXECUTABLE echo\n", encoding="utf-8")
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS 1
        INSTALL_JOB job job
        FORWARD_MODEL job
        FORWARD_MODEL job(<MESSAGE>=HELLO)
        """
    )
    Path(test_config_file_name).write_text(test_config_contents, encoding="utf-8")

    assert [
        (j.name, len(j.private_args))
        for j in ErtConfig.from_file(test_config_file_name).forward_model_steps
    ] == [("job", 0), ("job", 1)]


@pytest.mark.parametrize("eclipse_v", ["ECLIPSE100", "ECLIPSE300"])
def test_that_eclipse_fm_step_require_explicit_version(eclipse_v, plugins_ert_config):
    with (
        pytest.raises(
            ConfigValidationError,
            match=f".*Forward model step {eclipse_v} must"
            rf" be given a VERSION argument.*",
        ),
    ):
        _ = plugins_ert_config.from_file_contents(
            f"""
            NUM_REALIZATIONS  1
            FORWARD_MODEL {eclipse_v}
            """
        )


@pytest.mark.skipif(shutil.which("eclrun") is None, reason="eclrun is not in $PATH")
@pytest.mark.parametrize("eclipse_v", ["ECLIPSE100", "ECLIPSE300"])
def test_that_eclipse_fm_step_check_version_availability(eclipse_v, plugins_ert_config):
    with pytest.raises(
        ConfigValidationError,
        match=rf".*Unavailable {eclipse_v} version dummy."
        rf" Available versions: \[\'20.*",
    ):
        plugins_ert_config.from_file_contents(
            f"NUM_REALIZATIONS 1\nFORWARD_MODEL {eclipse_v}(<VERSION>=dummy)\n"
        )


@pytest.mark.parametrize("eclipse_v", ["ECLIPSE100", "ECLIPSE300"])
@pytest.mark.usefixtures("use_tmpdir")
def test_that_we_can_point_to_a_custom_eclrun_when_checking_versions(
    eclipse_v, plugins_ert_config
):
    eclrun_bin = Path("bin/eclrun")
    eclrun_bin.parent.mkdir()
    eclrun_bin.write_text("#!/bin/sh\necho 2036.1 2036.2 2037.1", encoding="utf-8")
    eclrun_bin.chmod(eclrun_bin.stat().st_mode | stat.S_IEXEC)
    config_file_name = "test.ert"
    Path(config_file_name).write_text(
        dedent(
            f"""
            NUM_REALIZATIONS 1
            SETENV ECLRUN_PATH {eclrun_bin.absolute().parent}
            FORWARD_MODEL {eclipse_v}(<VERSION>=2034.1)"""
        ),
        encoding="utf-8",
    )
    with (
        pytest.raises(
            ConfigValidationError,
            match=(
                rf".*Unavailable {eclipse_v} version 2034.1. "
                rf"Available versions: \[\'2036.1.*"
            ),
        ),
    ):
        plugins_ert_config.from_file(config_file_name)


@pytest.mark.skipif(shutil.which("eclrun") is not None, reason="eclrun is present")
@pytest.mark.parametrize("eclipse_v", ["ECLIPSE100", "ECLIPSE300"])
def test_that_no_error_thrown_when_checking_eclipse_version_and_eclrun_is_not_present(
    eclipse_v, plugins_ert_config
):
    _ = plugins_ert_config.from_file_contents(
        f"NUM_REALIZATIONS 1\nFORWARD_MODEL {eclipse_v}(<VERSION>=1)\n"
    )


def test_that_flow_fm_step_does_not_need_explicit_version(plugins_ert_config):
    plugins_ert_config.from_file_contents("NUM_REALIZATIONS 1\nFORWARD_MODEL FLOW\n")


def test_that_flow_fm_step_always_allow_explicit_default_version(plugins_ert_config):
    plugins_ert_config.from_file_contents(
        "NUM_REALIZATIONS 1\nFORWARD_MODEL FLOW(<VERSION>=default)\n"
    )


@pytest.mark.integration_test
@pytest.mark.skipif(shutil.which("flowrun") is None, reason="flowrun is not in $PATH")
def test_that_flow_fm_step_check_version_availability(plugins_ert_config):
    with pytest.raises(
        ConfigValidationError,
        match=r".*Unavailable Flow version dummy. Available versions: \[\'.*",
    ):
        plugins_ert_config.from_file_contents(
            "NUM_REALIZATIONS 1\nFORWARD_MODEL FLOW(<VERSION>=dummy)\n"
        )


def test_that_flow_fm_gives_config_warning_on_unknown_options(plugins_ert_config):
    with pytest.warns(ConfigWarning, match=r".*Unknown option.*Flow: .*DUMMY.*"):
        plugins_ert_config.from_file_contents(
            "NUM_REALIZATIONS 1\nFORWARD_MODEL FLOW(<DUMMY>=moredummy)\n"
        )


def test_that_plugin_forward_models_are_installed(tmp_path):
    (tmp_path / "test.ert").write_text(
        dedent(
            """
        NUM_REALIZATIONS  1
        FORWARD_MODEL PluginForwardModel(<arg1>=hello,<arg2>=world,<arg3>=derpyderp)
        """
        )
    )

    class PluginForwardModel(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="PluginForwardModel",
                command=["something", "<arg1>", "-f", "<arg2>", "<arg3>"],
            )

        def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
            if set(self.private_args.keys()) != {"<arg1>", "<arg2>", "<arg3>"}:
                raise ForwardModelStepValidationError("Bad")

        def validate_pre_realization_run(
            self, fm_step_json: ForwardModelStepJSON
        ) -> ForwardModelStepJSON:
            return fm_step_json

    plugin_fm_instance = PluginForwardModel()
    ert_config = ErtConfig.with_plugins(
        ErtRuntimePlugins(
            installed_forward_model_steps={plugin_fm_instance.name: plugin_fm_instance}
        )
    ).from_file(tmp_path / "test.ert")

    first_fm = ert_config.forward_model_steps[0]

    expected_attrs = {
        "name": "PluginForwardModel",
        "executable": "something",
        "stdin_file": None,
        "stdout_file": "PluginForwardModel.stdout",
        "stderr_file": "PluginForwardModel.stderr",
        "start_file": None,
        "target_file": None,
        "error_file": None,
        "max_running_minutes": None,
        "min_arg": 0,
        "max_arg": 0,
        "arglist": ["<arg1>", "-f", "<arg2>", "<arg3>"],
        "required_keywords": [],
        "arg_types": [],
        "environment": {
            "_ERT_ITERATION_NUMBER": "<ITER>",
            "_ERT_REALIZATION_NUMBER": "<IENS>",
            "_ERT_RUNPATH": "<RUNPATH>",
        },
        "default_mapping": {},
        "private_args": {
            "<arg1>": "hello",
            "<arg2>": "world",
            "<arg3>": "derpyderp",
        },
    }

    for a, v in expected_attrs.items():
        assert getattr(first_fm, a) == v, (
            f"Expected fm[{a}] to be {v} but was {getattr(first_fm, a)}"
        )

    fm_json = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        user_config_file=ert_config.user_config_file,
        run_id="some_id",
        iens=0,
        itr=0,
    )

    assert len(fm_json["jobList"]) == 1
    job_from_joblist = fm_json["jobList"][0]
    assert job_from_joblist["name"] == "PluginForwardModel"
    assert job_from_joblist["executable"] == "something"
    assert job_from_joblist["stdout"] == "PluginForwardModel.stdout.0"
    assert job_from_joblist["stderr"] == "PluginForwardModel.stderr.0"
    assert job_from_joblist["argList"] == ["hello", "-f", "world", "derpyderp"]


def test_that_plugin_forward_model_validation_failure_propagates(tmp_path):
    (tmp_path / "test.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS  1
            FORWARD_MODEL PluginFM(<arg1>=hello,<arg2>=world,<arg3>=derpyderp)
            """
        )
    )

    class FM(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="PluginFM",
                command=["something", "<arg1>", "-f", "<arg2>", "<arg3>"],
            )

        def validate_pre_realization_run(
            self, fm_json: ForwardModelStepJSON
        ) -> ForwardModelStepJSON:
            if fm_json["argList"][0] != "never":
                raise ForwardModelStepValidationError("Oh no")

            return fm_json

    ert_config = ErtConfig.with_plugins(
        ErtRuntimePlugins(installed_forward_model_steps={"PluginFM": FM()})
    ).from_file(tmp_path / "test.ert")

    first_fm = ert_config.forward_model_steps[0]
    with pytest.raises(ForwardModelStepValidationError, match="Oh no"):
        first_fm.validate_pre_realization_run({"argList": ["not hello"]})

    with pytest.raises(
        ConfigValidationError, match="Validation failed for forward model step"
    ):
        create_forward_model_json(
            context=ert_config.substitutions,
            forward_model_steps=ert_config.forward_model_steps,
            env_vars=ert_config.env_vars,
            user_config_file=ert_config.user_config_file,
            run_id="id",
            iens=0,
            itr=0,
        )


def test_that_plugin_forward_model_validation_accepts_valid_args(tmp_path):
    (tmp_path / "test.ert").write_text(
        dedent(
            """
        NUM_REALIZATIONS  1
        FORWARD_MODEL FM(<arg1>=never,<arg2>=world,<arg3>=derpyderp)
        """
        )
    )

    class FM(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="FM",
                command=["something", "<arg1>", "-f", "<arg2>", "<arg3>"],
            )

        def validate_pre_realization_run(
            self, fm_json: ForwardModelStepJSON
        ) -> ForwardModelStepJSON:
            if fm_json["argList"][0] != "never":
                raise ForwardModelStepValidationError("Oh no")

            return fm_json

    ert_config = ErtConfig.with_plugins(
        ErtRuntimePlugins(installed_forward_model_steps={"FM": FM()})
    ).from_file(tmp_path / "test.ert")
    first_fm = ert_config.forward_model_steps[0]

    first_fm.validate_pre_realization_run({"argList": ["never"]})

    create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        user_config_file=ert_config.user_config_file,
        run_id="id",
        iens=0,
        itr=0,
    )


def test_that_plugin_forward_model_raises_pre_realization_validation_error():
    class FM1(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="FM1",
                command=["the_executable.sh"],
            )

        def validate_pre_realization_run(
            self, fm_step_json: ForwardModelStepJSON
        ) -> ForwardModelStepJSON:
            raise ForwardModelStepValidationError(
                "This is a bad forward model step, don't use it"
            )

    class FM2(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="FM2",
                command=["something", "<arg1>", "-f", "<arg2>", "<arg3>"],
            )

        def validate_pre_realization_run(
            self, fm_json: ForwardModelStepJSON
        ) -> ForwardModelStepJSON:
            if fm_json["argList"][0] != "never":
                raise ForwardModelStepValidationError("Oh no")

            return fm_json

    config = ErtConfig.with_plugins(
        ErtRuntimePlugins(installed_forward_model_steps={"FM1": FM1(), "FM2": FM2()})
    ).from_file_contents(
        """
            NUM_REALIZATIONS  1
            FORWARD_MODEL FM1(<arg1>=never,<arg2>=world,<arg3>=derpyderp)
            FORWARD_MODEL FM2
            """
    )
    assert isinstance(config.forward_model_steps[0], FM1)
    assert config.forward_model_steps[0].name == "FM1"

    assert isinstance(config.forward_model_steps[1], FM2)
    assert config.forward_model_steps[1].name == "FM2"

    with pytest.raises(
        ConfigValidationError,
        match=r".*This is a bad forward model step, don't use it.*",
    ):
        create_forward_model_json(
            context=config.substitutions,
            forward_model_steps=config.forward_model_steps,
            env_vars=config.env_vars,
            user_config_file=config.user_config_file,
            run_id="id",
            iens=0,
            itr=0,
        )


def test_that_plugin_forward_model_raises_pre_experiment_validation_error_early():
    class InvalidFightingStyle(ForwardModelStepValidationError):
        pass

    class FM1(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(name="FM1", command=["the_executable.sh"])

        def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
            if self.name != "FM1":
                raise ForwardModelStepValidationError("Expected name to be FM1")

            raise InvalidFightingStyle("I don't think I wanna do hamster style anymore")

    class FM2(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="FM2",
                command=["the_executable.sh"],
            )

        def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
            if self.name != "FM2":
                raise ForwardModelStepValidationError("Expected name to be FM2")

            raise ForwardModelStepValidationError("well that's nice")

    with pytest.raises(ConfigValidationError, match=r".*hamster style.*that's nice.*"):
        _ = ErtConfig.with_plugins(
            ErtRuntimePlugins(
                installed_forward_model_steps={"FM1": FM1(), "FM2": FM2()}
            )
        ).from_file_contents(
            """
            NUM_REALIZATIONS 1
            FORWARD_MODEL FM1(<arg1>=never,<arg2>=world,<arg3>=derpyderp)
            FORWARD_MODEL FM2
            """
        )


def test_that_pre_run_substitution_forward_model_json_is_created_for_plugin_fms():
    class FM1(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="FM1",
                command=[
                    "the_executable.sh",
                    "sed",
                    "-i",
                    "<yo>",
                    "-c",
                    "<dawg>",
                    "<iherdulike>",
                    "<some_var>",
                    "iter",
                ],
            )

        def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
            assert fm_step_json["argList"] == [
                "sed",
                "-i",
                "dear",
                "-c",
                "good",
                "solonius",
                "schmidt",
                "iter",
            ]

            # It is in the arglist, but not in the forward model(...) invocation in the
            # ert config. Thus it is not a "private" arg in that sense.
            assert "<some_var>" not in self.private_args

            assert dict(self.private_args) == {
                "<arg1>": "dear",
                "<arg2>": "good",
                "<arg3>": "solonius",
            }

    ErtConfig.with_plugins(
        ErtRuntimePlugins(installed_forward_model_steps={"FM1": FM1()})
    ).from_file_contents(
        """
        NUM_REALIZATIONS  1

        DEFINE <yo> dear
        DEFINE <dawg> good
        DEFINE <iherdulike> solonius
        DEFINE <some_var> schmidt

        FORWARD_MODEL FM1(<arg1>=<yo>,<arg2>=<dawg>,<arg3>=<iherdulike>)
        """
    )


def test_that_plugin_forward_model_unexpected_errors_show_as_warnings():
    class FMWithAssertionError(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(name="FMWithAssertionError", command=["the_executable.sh"])

        def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
            raise AssertionError("I should be a warning")

    class FMWithFMStepValidationError(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="FMWithFMStepValidationError",
                command=["the_executable.sh"],
            )

        def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
            raise ForwardModelStepValidationError("I should not be a warning")

    with (
        pytest.raises(ConfigValidationError, match="I should not be a warning"),
        pytest.warns(ConfigWarning, match="I should be a warning"),
    ):
        _ = ErtConfig.with_plugins(
            ErtRuntimePlugins(
                installed_forward_model_steps={
                    "FMWithAssertionError": FMWithAssertionError(),
                    "FMWithFMStepValidationError": FMWithFMStepValidationError(),
                }
            )
        ).from_file_contents(
            """
            NUM_REALIZATIONS  1
            FORWARD_MODEL FMWithAssertionError(<arg1>=never,<arg2>=world,\
               <arg3>=derpyderp)
            FORWARD_MODEL FMWithFMStepValidationError
            """
        )


def test_that_plugin_fm_step_raises_validation_error_on_missing_required_argument():
    class FMThatNeedsABC(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="FMWithAssertionError",
                command=["echo", "<A>", "<B>", "<C>"],
                required_keywords=["A", "B", "C"],
            )

    with pytest.raises(
        ConfigValidationError,
        match="Required keyword C not found for forward model step FMThatNeedsABC",
    ):
        _ = ErtConfig.with_plugins(
            ErtRuntimePlugins(
                installed_forward_model_steps={
                    "FMThatNeedsABC": FMThatNeedsABC(),
                }
            )
        ).from_file_contents(
            """
            NUM_REALIZATIONS  1
            FORWARD_MODEL FMThatNeedsABC(A=never,B=world,\
               D=derpyderp)
            """
        )


def test_that_plugin_fm_step_is_parsed_successfully_with_required_arguments():
    class FMThatNeedsABC(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="FMWithAssertionError",
                command=["echo", "<A>", "<B>", "<C>"],
                required_keywords=["A", "B", "C"],
            )

    ErtConfig.with_plugins(
        ErtRuntimePlugins(
            installed_forward_model_steps={
                "FMThatNeedsABC": FMThatNeedsABC(),
            }
        )
    ).from_file_contents(
        """
        NUM_REALIZATIONS  1
        FORWARD_MODEL FMThatNeedsABC(A=never,B=world,\
           C=derpyderp)
        """
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_one_required_keyword_in_forward_model_is_validated():
    Path("step").write_text("EXECUTABLE echo\nREQUIRED MESSAGE", encoding="utf-8")
    with pytest.raises(
        ConfigValidationError, match="Required keyword MESSAGE not found"
    ):
        ErtConfig.from_file_contents(
            """
           NUM_REALIZATIONS 1
           INSTALL_JOB step step
           FORWARD_MODEL step
           """
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_all_required_keywords_in_forward_model_are_validated():
    Path("step").write_text(
        "EXECUTABLE echo\nREQUIRED MESSAGE1 MESSAGE2", encoding="utf-8"
    )
    with pytest.raises(
        ConfigValidationError, match="Required keywords MESSAGE1, MESSAGE2 not found"
    ):
        ErtConfig.from_file_contents(
            """
           NUM_REALIZATIONS 1
           INSTALL_JOB step step
           FORWARD_MODEL step
           """
        )


def test_that_site_fm_step_serializes_as_reference_to_site_plugin(use_tmpdir):
    class SiteForwardModel(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="SITE_FM",
                command=["echo", "helloworld", "<ONE>", "<TWO>"],
            )

    with use_runtime_plugins(
        ErtRuntimePlugins(installed_forward_model_steps={"SITE_FM": SiteForwardModel()})
    ):
        site_fm = SiteForwardModel()
        # PS: It is missing private args, this must be added by mutation
        # as it is the current practice in ert config
        site_fm.private_args = {"ONE": 1, "TWO": "2"}

        serialized_fm_step = site_fm.model_dump(mode="json")
        assert serialized_fm_step == {
            "name": "SITE_FM",
            "type": "site_installed",
            "private_args": {"ONE": 1, "TWO": "2"},
        }


def test_that_site_fm_step_deserialization_is_overwritten_by_site_installed_fmsteps(
    use_tmpdir,
):
    custom_echo_path = Path("custom_echo.sh")
    custom_echo_path.write_text("#!/bin/bash\necho hello", encoding="utf-8")
    custom_echo_path.chmod(custom_echo_path.stat().st_mode | stat.S_IEXEC)

    class SomeUpdatedForwardModel(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="SITE_FM",
                command=["custom_echo.sh", "helloworld", "<hello>", "<hi>", "<noop>"],
            )

    with use_runtime_plugins(
        ErtRuntimePlugins(
            installed_forward_model_steps={"SITE_FM": SomeUpdatedForwardModel()}
        )
    ):
        site_fm_with_updated_executable = TypeAdapter(
            SiteOrUserForwardModelStep
        ).validate_python(
            {
                "name": "SITE_FM",
                "type": "site_installed",
                "private_args": {"hello": "hi", "hi": "2"},
            }
        )

        assert isinstance(
            site_fm_with_updated_executable, SiteInstalledForwardModelStep
        )
        assert site_fm_with_updated_executable.executable == "custom_echo.sh"
        assert site_fm_with_updated_executable.private_args == {
            "hello": "hi",
            "hi": "2",
        }


def test_that_user_fm_step_retains_executable_and_private_args_when_serialized(
    use_tmpdir,
):
    Path("fm_step").write_text("EXECUTABLE echo\nARGLIST ARG1 ARG2", encoding="utf-8")
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS 1
        INSTALL_JOB user_fm fm_step
        FORWARD_MODEL user_fm(ARG1=<arg1>,ARG2=<arg2>)
        """
    )
    Path("config.ert").write_text(test_config_contents, encoding="utf-8")

    ert_config = ErtConfig.from_file("config.ert")
    user_fm = ert_config.forward_model_steps[0]
    assert user_fm.type == "user_installed"
    assert user_fm.executable == "echo"
    assert user_fm.arglist == ["ARG1", "ARG2"]
    assert user_fm.private_args == {"ARG1": "<arg1>", "ARG2": "<arg2>"}
    assert (
        TypeAdapter(SiteOrUserForwardModelStep).validate_python(user_fm.model_dump())
        == user_fm
    )


def test_that_fm_step_serializes_name_and_private_args_only_for_site_and_full_for_user(
    use_tmpdir,
):
    Path("fm_step").write_text(
        "EXECUTABLE echo\nARGLIST ONE TWO THREE", encoding="utf-8"
    )
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS 1
        INSTALL_JOB users_fm fm_step
        FORWARD_MODEL users_fm(ONE=1,TWO=2,THREE=3)
        FORWARD_MODEL SITE_INSTALLED_FM(ONE=1,TWO=2,THREE=3)
        """
    )
    Path("config.ert").write_text(test_config_contents, encoding="utf-8")

    class SiteForwardModel(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="SITE_INSTALLED_FM",
                command=["echo", "hello_from_site", "<ONE>", "<TWO>", "<THREE>"],
            )

    site_plugins = ErtRuntimePlugins(
        installed_forward_model_steps={"SITE_INSTALLED_FM": SiteForwardModel()}
    )
    ert_config = ErtConfig.with_plugins(site_plugins).from_file("config.ert")
    [user_fm, site_fm] = ert_config.forward_model_steps
    assert user_fm.type == "user_installed"
    assert site_fm.model_dump() == {
        "type": "site_installed",
        "name": "SITE_INSTALLED_FM",
        "private_args": {"ONE": "1", "TWO": "2", "THREE": "3"},
    }

    assert user_fm.private_args == {"ONE": "1", "TWO": "2", "THREE": "3"}
    assert user_fm.private_args == site_fm.private_args


def test_that_deserializing_site_forward_model_without_site_plugins_raises_error():
    with pytest.raises(KeyError, match="Trying to find site-installed forward model"):
        SiteInstalledForwardModelStep.model_validate(
            {"type": "site_installed", "name": "queue pool's closed"}
        )


def test_that_deserializing_site_forward_model_step_not_in_plugins_raises_error():
    with (
        pytest.raises(
            KeyError,
            match=r"Expected forward model step queue pool",
        ),
        use_runtime_plugins(ErtRuntimePlugins(installed_forward_model_steps={})),
    ):
        SiteInstalledForwardModelStep.model_validate(
            {"type": "site_installed", "name": "queue pool's closed"}
        )
