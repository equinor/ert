import os
import os.path
import stat
from textwrap import dedent

import pytest

from ert.config import ConfigValidationError, ConfigWarning, ExtJob
from ert.config.parsing import SchemaItemType


@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model_raises_on_missing():
    with pytest.raises(ConfigValidationError, match="No such file or directory"):
        _ = ExtJob.from_config_file("CONFIG_FILE")


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
    job = ExtJob.from_config_file("CONFIG")
    assert job.name == "CONFIG"
    assert job.stdout_file is None
    assert job.stderr_file is None

    assert job.executable == os.path.join(os.getcwd(), "script.sh")
    assert os.access(job.executable, os.X_OK)

    assert job.min_arg is None

    job = ExtJob.from_config_file("CONFIG", name="Job")
    assert job.name == "Job"
    assert repr(job).startswith("ExtJob(")


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
    job = ExtJob.from_config_file("CONFIG")
    assert job.min_arg == 2
    assert job.max_arg == 7
    argTypes = job.arg_types
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
        _ = ExtJob.from_config_file("CONFIG")


@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model_missing_raises():
    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write("EXECUTABLE missing_script.sh\n")
    with pytest.raises(ConfigValidationError, match="Could not find executable"):
        _ = ExtJob.from_config_file("CONFIG")


@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model_execu_missing_raises():
    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write("EXECU missing_script.sh\n")
    with pytest.raises(ConfigValidationError, match="EXECUTABLE must be set"):
        _ = ExtJob.from_config_file("CONFIG")


@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model_is_directory_raises():
    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write("EXECUTABLE /tmp\n")
    with pytest.raises(ConfigValidationError, match="directory"):
        _ = ExtJob.from_config_file("CONFIG")


@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model_foreign_raises():
    with open("CONFIG", "w", encoding="utf-8") as f:
        f.write("EXECUTABLE /etc/passwd\n")
    with pytest.raises(ConfigValidationError, match="File not executable"):
        _ = ExtJob.from_config_file("CONFIG")


def test_ext_job_optionals(
    tmp_path,
):
    executable = tmp_path / "exec"
    executable.write_text("")
    st = os.stat(executable)
    os.chmod(executable, st.st_mode | stat.S_IEXEC)
    config_file = tmp_path / "config_file"
    config_file.write_text("EXECUTABLE exec\n")
    ext_job = ExtJob.from_config_file(str(config_file))
    assert ext_job.name == "config_file"


@pytest.mark.usefixtures("use_tmpdir")
def test_ext_job_env_and_exec_env_is_set():
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
    ext_job = ExtJob.from_config_file("CONFIG")

    assert ext_job.environment["a"] == "b"
    assert ext_job.environment["c"] == "d"

    assert ext_job.exec_env["a1"] == "b1"
    assert ext_job.exec_env["c1"] == "d1"


@pytest.mark.usefixtures("use_tmpdir")
def test_ext_job_stdout_stderr_defaults_to_filename():
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

    ext_job = ExtJob.from_config_file("CONFIG")

    assert ext_job.name == "CONFIG"
    assert ext_job.stdout_file == "CONFIG.stdout"
    assert ext_job.stderr_file == "CONFIG.stderr"


@pytest.mark.usefixtures("use_tmpdir")
def test_ext_job_stdout_stderr_null_results_in_none():
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

    ext_job = ExtJob.from_config_file("CONFIG")

    assert ext_job.name == "CONFIG"
    assert ext_job.stdin_file is None
    assert ext_job.stdout_file is None
    assert ext_job.stderr_file is None


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

    ext_job = ExtJob.from_config_file("CONFIG")

    assert ext_job.arglist == ["<A>", "B", "<C>", "<D>", "<E>"]


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

    ext_job = ExtJob.from_config_file("CONFIG")
    assert ext_job.environment == ext_job.default_env


@pytest.mark.usefixtures("use_tmpdir")
def test_ext_job_arglist_with_weird_characters():
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

    ext_job = ExtJob.from_config_file("CONFIG")
    assert ext_job.environment == ext_job.default_env
    assert ext_job.arglist == ["-i", "s/^RUNSPEC.*/|RUNSPEC\nNOSIM/", "<ECLBASE>.DATA"]
