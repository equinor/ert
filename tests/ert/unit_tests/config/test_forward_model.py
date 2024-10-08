import logging
import os
import os.path
import shutil
import stat
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest
from hypothesis import given, settings

from ert.config import ConfigValidationError, ConfigWarning, ErtConfig
from ert.config.ert_config import _forward_model_step_from_config_file
from ert.config.forward_model_step import (
    ForwardModelStepJSON,
    ForwardModelStepPlugin,
    ForwardModelStepValidationError,
)
from ert.config.parsing import SchemaItemType
from ert.substitution_list import SubstitutionList

from .config_dict_generator import config_generators


@pytest.fixture()
def mock_eclrun():
    with open("eclrun", "w", encoding="utf-8") as f:
        f.write("""#!/usr/bin/env python\n\nprint("4 2 8")""")
    os.chmod("eclrun", os.stat("eclrun").st_mode | stat.S_IEXEC)
    old_path = os.environ["PATH"]
    eclrun_path = shutil.which("eclrun")
    if eclrun_path is None:
        os.environ["PATH"] = old_path + os.pathsep + os.getcwd()
    else:
        os.environ["PATH"] = old_path.replace(
            str(Path(eclrun_path).parent), os.getcwd()
        )
    yield
    os.environ["PATH"] = old_path


@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model_raises_on_missing():
    with pytest.raises(ConfigValidationError, match="No such file or directory"):
        _ = _forward_model_step_from_config_file("CONFIG_FILE")


@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model():
    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write("STDOUT null\n")
        f.write("STDERR null\n")
        f.write("EXECUTABLE script.sh\n")
    name = "script.sh"
    with open(name, "w", encoding="utf-8") as f:
        f.write("This is a script")
    mode = os.stat(name).st_mode
    mode |= stat.S_IXUSR | stat.S_IXGRP
    os.chmod(name, stat.S_IMODE(mode))
    fm_step = _forward_model_step_from_config_file("CONFIG")
    assert fm_step.name == "CONFIG"
    assert fm_step.stdout_file is None
    assert fm_step.stderr_file is None

    assert fm_step.executable == os.path.join(os.getcwd(), "script.sh")
    assert os.access(fm_step.executable, os.X_OK)

    assert fm_step.min_arg is None

    fm_step = _forward_model_step_from_config_file("CONFIG", name="Step")
    assert fm_step.name == "Step"
    assert repr(fm_step).startswith("ForwardModelStep(")


@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model_upgraded():
    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write("EXECUTABLE script.sh\n")
        f.write("MIN_ARG 2\n")
        f.write("MAX_ARG 7\n")
        f.write("ARG_TYPE 0 INT\n")
        f.write("ARG_TYPE 1 FLOAT\n")
        f.write("ARG_TYPE 2 STRING\n")
        f.write("ARG_TYPE 3 BOOL\n")
        f.write("ARG_TYPE 4 RUNTIME_FILE\n")
        f.write("ARG_TYPE 5 RUNTIME_INT\n")

    name = "script.sh"
    with open(name, "w", encoding="utf-8") as f:
        f.write("This is a script")
    mode = os.stat(name).st_mode
    mode |= stat.S_IXUSR | stat.S_IXGRP
    os.chmod(name, stat.S_IMODE(mode))
    fm_step = _forward_model_step_from_config_file("CONFIG")
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


@pytest.mark.usefixtures("use_tmpdir")
def test_portable_exe_error_message():
    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write("PORTABLE_EXE script.sh\n")

    name = "script.sh"
    with open(name, "w", encoding="utf-8") as f:
        f.write("This is a script")
        name = "script.sh"
        with open(name, "w", encoding="utf-8") as f:
            f.write("This is a script")
        mode = os.stat(name).st_mode
        mode |= stat.S_IXUSR | stat.S_IXGRP
        os.chmod(name, stat.S_IMODE(mode))
    with pytest.raises(
        ConfigValidationError, match="EXECUTABLE must be set"
    ), pytest.warns(ConfigWarning, match='"PORTABLE_EXE" key is deprecated'):
        _ = _forward_model_step_from_config_file("CONFIG")


@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model_missing_raises():
    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write("EXECUTABLE missing_script.sh\n")
    with pytest.raises(ConfigValidationError, match="Could not find executable"):
        _ = _forward_model_step_from_config_file("CONFIG")


@pytest.mark.filterwarnings("ignore:.*Unknown keyword 'EXECU'.*:UserWarning")
@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model_execu_missing_raises():
    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write("EXECU missing_script.sh\n")
    with pytest.raises(ConfigValidationError, match="EXECUTABLE must be set"):
        _ = _forward_model_step_from_config_file("CONFIG")


@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model_is_directory_raises():
    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write("EXECUTABLE /tmp\n")
    with pytest.raises(ConfigValidationError, match="directory"):
        _ = _forward_model_step_from_config_file("CONFIG")


@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model_foreign_raises():
    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write("EXECUTABLE /etc/passwd\n")
    with pytest.raises(ConfigValidationError, match="File not executable"):
        _ = _forward_model_step_from_config_file("CONFIG")


def test_forward_model_optionals(
    tmp_path,
):
    executable = tmp_path / "exec"
    executable.write_text("")
    st = os.stat(executable)
    os.chmod(executable, st.st_mode | stat.S_IEXEC)
    config_file = tmp_path / "config_file"
    config_file.write_text("EXECUTABLE exec\n")
    forward_model = _forward_model_step_from_config_file(str(config_file))
    assert forward_model.name == "config_file"


@pytest.mark.usefixtures("use_tmpdir")
def test_forward_model_env_and_exec_env_is_set():
    with open("exec", "w", encoding="utf-8") as f:
        pass

    os.chmod("exec", stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """
        EXECUTABLE exec
        ENV a b
        ENV c d
        EXEC_ENV a1 b1
        EXEC_ENV c1 d1
        """
            )
        )
    forward_model = _forward_model_step_from_config_file("CONFIG")

    assert forward_model.environment["a"] == "b"
    assert forward_model.environment["c"] == "d"

    assert forward_model.exec_env["a1"] == "b1"
    assert forward_model.exec_env["c1"] == "d1"


@pytest.mark.usefixtures("use_tmpdir")
def test_forward_model_stdout_stderr_defaults_to_filename():
    with open("exec", "w", encoding="utf-8") as f:
        pass

    os.chmod("exec", stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """
        EXECUTABLE exec
        """
            )
        )

    forward_model = _forward_model_step_from_config_file("CONFIG")

    assert forward_model.name == "CONFIG"
    assert forward_model.stdout_file == "CONFIG.stdout"
    assert forward_model.stderr_file == "CONFIG.stderr"


@pytest.mark.usefixtures("use_tmpdir")
def test_forward_model_stdout_stderr_null_results_in_none():
    with open("exec", "w", encoding="utf-8") as f:
        pass

    os.chmod("exec", stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """
        EXECUTABLE exec
        STDIN null
        STDOUT null
        STDERR null
        """
            )
        )

    forward_model = _forward_model_step_from_config_file("CONFIG")

    assert forward_model.name == "CONFIG"
    assert forward_model.stdin_file is None
    assert forward_model.stdout_file is None
    assert forward_model.stderr_file is None


@pytest.mark.usefixtures("use_tmpdir")
def test_that_arglist_is_parsed_correctly():
    with open("exec", "w", encoding="utf-8") as f:
        pass

    os.chmod("exec", stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """
        EXECUTABLE exec
        ARGLIST <A> B <C> <D> <E>
        """
            )
        )

    forward_model = _forward_model_step_from_config_file("CONFIG")

    assert forward_model.arglist == ["<A>", "B", "<C>", "<D>", "<E>"]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_default_env_is_set():
    with open("exec", "w", encoding="utf-8") as f:
        pass

    os.chmod("exec", stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """
        EXECUTABLE exec
        """
            )
        )

    forward_model = _forward_model_step_from_config_file("CONFIG")
    assert forward_model.environment == forward_model.default_env


@pytest.mark.usefixtures("use_tmpdir")
def test_forward_model_arglist_with_weird_characters():
    with open("exec", "w", encoding="utf-8") as f:
        pass

    os.chmod("exec", stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write(
            dedent(
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
        """
            )
        )

    forward_model = _forward_model_step_from_config_file("CONFIG")
    assert forward_model.environment == forward_model.default_env
    assert forward_model.arglist == [
        "-i",
        "s/^RUNSPEC.*/|RUNSPEC\nNOSIM/",
        "<ECLBASE>.DATA",
    ]


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


def test_that_substitutions_can_be_done_in_job_names():
    """
    Regression test for a usage case involving setting ECL100 or ECL300
    that was broken by changes to forward_model substitutions.
    """
    ert_config = ErtConfig.with_plugins().from_file_contents(
        """
        NUM_REALIZATIONS  1
        DEFINE <ECL100OR300> E100
        FORWARD_MODEL ECLIPS<ECL100OR300>(<VERSION>=2024.1, <NUM_CPU>=42, <OPTS>="-m")
        """
    )
    assert len(ert_config.forward_model_steps) == 1
    job = ert_config.forward_model_steps[0]
    assert job.name == "ECLIPSE100"


def test_parsing_forward_model_with_double_dash_is_possible():
    """This is a regression test, making sure that we can put double dashes in strings.
    The use case is that a file name is utilized that contains two consecutive hyphens,
    which by the ert config parser used to be interpreted as a comment. In the new
    parser this is allowed"""
    res_config = ErtConfig.with_plugins().from_file_contents(
        """
        NUM_REALIZATIONS  1
        JOBNAME job_%d--hei
        FORWARD_MODEL COPY_FILE(<FROM>=foo,<TO>=something/hello--there.txt)
        """
    )
    assert res_config.model_config.jobname_format_string == "job_<IENS>--hei"
    assert (
        res_config.forward_model_steps[0].private_args["<TO>"]
        == "something/hello--there.txt"
    )


def test_parsing_forward_model_with_quotes_does_not_introduce_spaces():
    """this is a regression test, making sure that we do not by mistake introduce
    spaces while parsing forward model lines that contain quotation marks

    the use case is that a file name is utilized that contains two consecutive hyphens,
    which by the ert config parser is interpreted as a comment - to circumvent the
    comment interpretation, quotation marks are used"""

    str_with_quotes = """smt/<foo>"/bar"/xx/"t--s.s"/yy/"z/z"/oo"""
    ert_config = ErtConfig.with_plugins().from_file_contents(
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


def test_that_comments_are_ignored():
    """This is a regression test, making sure that we can put double dashes in strings.
    The use case is that a file name is utilized that contains two consecutive hyphens,
    which by the ert config parser used to be interpreted as a comment. In the new
    parser this is allowed"""

    res_config = ErtConfig.with_plugins().from_file_contents(
        """
        NUM_REALIZATIONS  1
        --comment
        JOBNAME job_%d--hei --hei
        FORWARD_MODEL COPY_FILE(<FROM>=foo,<TO>=something/hello--there.txt)--foo
        """
    )
    assert res_config.model_config.jobname_format_string == "job_<IENS>--hei"
    assert (
        res_config.forward_model_steps[0].private_args["<TO>"]
        == "something/hello--there.txt"
    )


def test_that_quotations_in_forward_model_arglist_are_handled_correctly():
    """This is a regression test, making sure that quoted strings behave consistently.
    They should all result in the same.
    See https://github.com/equinor/ert/issues/2766"""

    res_config = ErtConfig.with_plugins().from_file_contents(
        """
        NUM_REALIZATIONS  1
        FORWARD_MODEL COPY_FILE(<FROM>='some, thing', <TO>="some stuff", <FILE>=file.txt)
        FORWARD_MODEL COPY_FILE(<FROM>='some, thing', <TO>='some stuff', <FILE>=file.txt)
        FORWARD_MODEL COPY_FILE(<FROM>="some, thing", <TO>="some stuff", <FILE>=file.txt)
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


def test_that_positional_forward_model_args_gives_config_validation_error():
    with pytest.raises(ConfigValidationError, match="Did not expect character: <"):
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
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.warns(ConfigWarning, match="Duplicate forward model step"):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_forward_model_substitution_does_not_warn_about_reaching_max_iterations(
    caplog,
):
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS 1
        FORWARD_MODEL ECLIPSE100(<VERSION>=2020.2)
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    ert_config = ErtConfig.with_plugins().from_file(test_config_file_name)
    with caplog.at_level(logging.WARNING):
        ert_config.forward_model_data_to_json(0, 0, 0)
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
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.warns(ConfigWarning, match="Duplicate forward model step"):
        _ = ErtConfig.from_file(test_config_file_name)


def test_that_spaces_in_forward_model_args_are_dropped():
    # Intentionally inserted several spaces before comma
    ert_config = ErtConfig.with_plugins().from_file_contents(
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
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    assert [
        (j.name, len(j.private_args))
        for j in ErtConfig.from_file(test_config_file_name).forward_model_steps
    ] == [("job", 0), ("job", 1)]


@pytest.mark.parametrize("eclipse_v", ["100", "300"])
def test_that_eclipse_jobs_require_version_field(eclipse_v):
    with pytest.raises(
        ConfigValidationError,
        match=f".*Forward model step ECLIPSE{eclipse_v} must be given a VERSION argument.*",
    ):
        _ = ErtConfig.with_plugins().from_file_contents(
            f"""
            NUM_REALIZATIONS  1
            FORWARD_MODEL ECLIPSE{eclipse_v}
            """
        )


@pytest.mark.parametrize("eclipse_v", ["100", "300"])
@pytest.mark.usefixtures("use_tmpdir")
def test_that_eclipse_jobs_check_version(eclipse_v, mock_eclrun):
    ecl100_config_file_name = "ecl100_config.yml"
    ecl300_config_file_name = "ecl300_config.yml"

    ecl100_config_content = f"eclrun_env:\n  PATH: {os.getcwd()}\n"
    ecl300_config_content = f"eclrun_env:\n  PATH: {os.getcwd()}\n"
    ert_config_contents = (
        f"NUM_REALIZATIONS  1\nFORWARD_MODEL ECLIPSE{eclipse_v} (<VERSION>=1)\n"
    )

    # Write config file
    config_file_name = "test.ert"
    Path(config_file_name).write_text(ert_config_contents, encoding="utf-8")
    # Write ecl100_config file
    Path(ecl100_config_file_name).write_text(ecl100_config_content, encoding="utf-8")
    # Write ecl300_config file
    Path(ecl300_config_file_name).write_text(ecl300_config_content, encoding="utf-8")
    with patch(
        "ert.plugins.hook_implementations.forward_model_steps.ErtPluginManager"
    ) as mock:
        instance = mock.return_value
        instance.get_ecl100_config_path.return_value = ecl100_config_file_name
        instance.get_ecl300_config_path.return_value = ecl300_config_file_name
        with pytest.raises(
            ConfigValidationError,
            match=rf".*Unavailable ECLIPSE{eclipse_v} version 1 current supported versions \['4', '2', '8'\].*",
        ):
            _ = ErtConfig.with_plugins().from_file(config_file_name)


@pytest.mark.skipif(shutil.which("eclrun") is not None, reason="eclrun is available")
@pytest.mark.parametrize("eclipse_v", ["100", "300"])
@pytest.mark.usefixtures("use_tmpdir")
def test_that_no_error_thrown_when_checking_eclipse_version_and_eclrun_is_not_present(
    eclipse_v,
):
    _ = ErtConfig.with_plugins().from_file_contents(
        f"NUM_REALIZATIONS  1\nFORWARD_MODEL ECLIPSE{eclipse_v} (<VERSION>=1)\n"
    )


@pytest.mark.skipif(shutil.which("eclrun") is not None, reason="eclrun is available")
@pytest.mark.parametrize("eclipse_v", ["100", "300"])
@pytest.mark.usefixtures("use_tmpdir")
def test_that_no_error_thrown_when_checking_eclipse_version_and_no_ecl_config_defined(
    eclipse_v, mock_eclrun
):
    ert_config_contents = (
        f"NUM_REALIZATIONS  1\nFORWARD_MODEL ECLIPSE{eclipse_v} (<VERSION>=1)\n"
    )

    # Write config file
    config_file_name = "test.ert"
    Path(config_file_name).write_text(ert_config_contents, encoding="utf-8")
    with patch(
        "ert.plugins.hook_implementations.forward_model_steps.ErtPluginManager"
    ) as mock:
        instance = mock.return_value
        instance.get_ecl100_config_path.return_value = None
        instance.get_ecl300_config_path.return_value = None
        _ = ErtConfig.with_plugins().from_file(config_file_name)


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
        def __init__(self):
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

    config = ErtConfig.with_plugins(
        forward_model_step_classes=[PluginForwardModel]
    ).from_file(tmp_path / "test.ert")

    first_fm = config.forward_model_steps[0]

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
        "exec_env": {},
        "default_mapping": {},
        "private_args": SubstitutionList(
            {
                "<arg1>": "hello",
                "<arg2>": "world",
                "<arg3>": "derpyderp",
            }
        ),
    }

    for a, v in expected_attrs.items():
        assert (
            getattr(first_fm, a) == v
        ), f"Expected fm[{a}] to be {v} but was {getattr(first_fm,a)}"

    fm_json = config.forward_model_data_to_json("some_id", 0, 0)
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
        def __init__(self):
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

    config = ErtConfig.with_plugins(forward_model_step_classes=[FM]).from_file(
        tmp_path / "test.ert"
    )

    first_fm = config.forward_model_steps[0]
    with pytest.raises(ForwardModelStepValidationError, match="Oh no"):
        first_fm.validate_pre_realization_run({"argList": ["not hello"]})

    with pytest.raises(
        ConfigValidationError, match="Validation failed for forward model step"
    ):
        _ = config.forward_model_data_to_json("id", 0, 0)


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
        def __init__(self):
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

    config = ErtConfig.with_plugins(forward_model_step_classes=[FM]).from_file(
        tmp_path / "test.ert"
    )
    first_fm = config.forward_model_steps[0]

    first_fm.validate_pre_realization_run({"argList": ["never"]})

    _ = config.forward_model_data_to_json("id", 0, 0)


def test_that_plugin_forward_model_raises_pre_realization_validation_error(tmp_path):
    (tmp_path / "test.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS  1
            FORWARD_MODEL FM1(<arg1>=never,<arg2>=world,<arg3>=derpyderp)
            FORWARD_MODEL FM2
            """
        )
    )

    class FM1(ForwardModelStepPlugin):
        def __init__(self):
            super().__init__(
                name="FM1",
                command=["the_executable.sh"],
            )

        def validate_pre_realization_run(
            self, fm_step_json: ForwardModelStepJSON
        ) -> ForwardModelStepJSON:
            raise ForwardModelStepValidationError(
                "This is a bad forward model step, dont use it"
            )

    class FM2(ForwardModelStepPlugin):
        def __init__(self):
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

    config = ErtConfig.with_plugins(forward_model_step_classes=[FM1, FM2]).from_file(
        tmp_path / "test.ert"
    )
    assert isinstance(config.forward_model_steps[0], FM1)
    assert config.forward_model_steps[0].name == "FM1"

    assert isinstance(config.forward_model_steps[1], FM2)
    assert config.forward_model_steps[1].name == "FM2"

    with pytest.raises(
        ConfigValidationError,
        match=".*This is a bad forward model step, dont use it.*",
    ):
        config.forward_model_data_to_json("id", 0, 0)


def test_that_plugin_forward_model_raises_pre_experiment_validation_error_early(
    tmp_path,
):
    (tmp_path / "test.ert").write_text(
        """
        NUM_REALIZATIONS  1
        FORWARD_MODEL FM1(<arg1>=never,<arg2>=world,<arg3>=derpyderp)
        FORWARD_MODEL FM2
        """
    )

    class InvalidFightingStyle(ForwardModelStepValidationError):
        pass

    class FM1(ForwardModelStepPlugin):
        def __init__(self):
            super().__init__(name="FM1", command=["the_executable.sh"])

        def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
            if self.name != "FM1":
                raise ForwardModelStepValidationError("Expected name to be FM1")

            raise InvalidFightingStyle("I don't think I wanna do hamster style anymore")

    class FM2(ForwardModelStepPlugin):
        def __init__(self):
            super().__init__(
                name="FM2",
                command=["the_executable.sh"],
            )

        def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
            if self.name != "FM2":
                raise ForwardModelStepValidationError("Expected name to be FM2")

            raise ForwardModelStepValidationError("well that's nice")

    with pytest.raises(ConfigValidationError, match=".*hamster style.*that's nice.*"):
        _ = ErtConfig.with_plugins(forward_model_step_classes=[FM1, FM2]).from_file(
            tmp_path / "test.ert"
        )


def test_that_pre_run_substitution_forward_model_json_is_created_for_plugin_fms(
    tmp_path,
):
    (tmp_path / "test.ert").write_text(
        dedent(
            """
        NUM_REALIZATIONS  1

        DEFINE <yo> dear
        DEFINE <dawg> good
        DEFINE <iherdulike> solonius
        DEFINE <some_var> schmidt

        FORWARD_MODEL FM1(<arg1>=<yo>,<arg2>=<dawg>,<arg3>=<iherdulike>)
        """
        )
    )

    class FM1(ForwardModelStepPlugin):
        def __init__(self):
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

    ErtConfig.with_plugins(forward_model_step_classes=[FM1]).from_file(
        tmp_path / "test.ert"
    )


def test_that_plugin_forward_model_unexpected_errors_show_as_warnings(tmp_path):
    (tmp_path / "test.ert").write_text(
        """
        NUM_REALIZATIONS  1
        FORWARD_MODEL FMWithAssertionError(<arg1>=never,<arg2>=world,<arg3>=derpyderp)
        FORWARD_MODEL FMWithFMStepValidationError
        """
    )

    class FMWithAssertionError(ForwardModelStepPlugin):
        def __init__(self):
            super().__init__(name="FMWithAssertionError", command=["the_executable.sh"])

        def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
            raise AssertionError("I should be a warning")

    class FMWithFMStepValidationError(ForwardModelStepPlugin):
        def __init__(self):
            super().__init__(
                name="FMWithFMStepValidationError",
                command=["the_executable.sh"],
            )

        def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
            raise ForwardModelStepValidationError("I should not be a warning")

    with pytest.raises(
        ConfigValidationError, match="I should not be a warning"
    ), pytest.warns(ConfigWarning, match="I should be a warning"):
        _ = ErtConfig.with_plugins(
            forward_model_step_classes=[
                FMWithFMStepValidationError,
                FMWithAssertionError,
            ]
        ).from_file(tmp_path / "test.ert")
