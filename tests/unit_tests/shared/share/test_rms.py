import json
import os
import pkgutil
import shutil
import stat
import subprocess
import sys
from os.path import dirname
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from tests.utils import SOURCE_DIR

from ._import_from_location import import_from_location

if TYPE_CHECKING:
    from importlib.abc import FileLoader


# import rms.py from ert/forward-models/res/script
# package-data path which. These are kept out of the ert package to avoid the
# overhead of importing ert. This is necessary as these may be invoked as a
# subprocess on each realization.
rms = import_from_location(
    "rms",
    os.path.join(
        SOURCE_DIR, "src/ert/shared/share/ert/forward-models/res/script/rms.py"
    ),
)


rms_run = rms.run

TEST_ENV_WRAPPER = """\
#!/usr/bin/env bash
PATH_PREFIX_EXPECTED={expected_path_prefix}
if [[ $PATH_PREFIX != $PATH_PREFIX_EXPECTED ]]
then
    echo "PATH_PREFIX set incorrectly"
    echo $PATH_PREFIX should be $PATH_PREFIX_EXPECTED
    exit 1
fi
PYPATH_EXPECTED={expected_pythonpath}
# first user defined, then config defined, then rest:
if [[ $PYTHONPATH != $PYPATH_EXPECTED ]]
then
    echo "PYTHONPATH set incorrectly"
    echo $PYTHONPATH should be $PYPATH_EXPECTED
    exit 1
fi
$@
"""


def _get_ert_shared_dir():
    ert_shared_loader = cast("FileLoader", pkgutil.get_loader("ert.shared"))
    return dirname(ert_shared_loader.get_filename())


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "test_input,expected_result",
    [
        (0, 422851785),
        (1, 723121249),
        (2, 132312123),
    ],
)
def test_run_class_multi_seed(monkeypatch, test_input, expected_result, source_root):
    with open("rms_config.yml", "w", encoding="utf-8") as f:
        f.write(f"executable:  {os.getcwd()}/bin/rms")

    os.mkdir("test_run_multi_seed")
    os.mkdir("run_path")
    os.mkdir("bin")
    os.mkdir("project")
    shutil.copy(os.path.join(source_root, "tests/unit_tests/shared/share/rms"), "bin")
    monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")

    action = {"exit_status": 0}
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    seed_file = ["3", "422851785", "723121249", "132312123"]
    with open("run_path/random.seeds", "w", encoding="utf-8") as f:
        f.write("\n".join(seed_file))

    r = rms.RMSRun(test_input, "project", "workflow", run_path="run_path")
    assert r.seed == expected_result


@pytest.mark.usefixtures("use_tmpdir")
def test_create():
    with pytest.raises(OSError):
        rms.RMSRun(0, "/project/does/not/exist", "workflow")

        os.mkdir("rms")
        rms.RMSRun(0, "rms", "workflow")


@pytest.mark.usefixtures("use_tmpdir")
def test_run_class(monkeypatch, source_root):
    with open("rms_config.yml", "w", encoding="utf-8") as f:
        f.write(f"executable:  {os.getcwd()}/bin/rms")

    os.mkdir("run_path")
    os.mkdir("bin")
    os.mkdir("project")
    shutil.copy(os.path.join(source_root, "tests/unit_tests/shared/share/rms"), "bin")
    monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")

    action = {"exit_status": 0}
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    r = rms.RMSRun(0, "project", "workflow", run_path="run_path", allow_no_env=True)
    r.run()

    action = {"exit_status": 1}
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    r = rms.RMSRun(0, "project", "workflow", run_path="run_path", allow_no_env=True)
    with pytest.raises(rms.RMSRunException):
        r.run()

    action = {"exit_status": 0}
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    r = rms.RMSRun(
        0,
        "project",
        "workflow",
        run_path="run_path",
        target_file="some_file",
        allow_no_env=True,
    )
    with pytest.raises(rms.RMSRunException):
        r.run()

    action = {
        "exit_status": 0,
        "target_file": os.path.join(os.getcwd(), "some_file"),
    }
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    r = rms.RMSRun(
        0,
        "project",
        "workflow",
        run_path="run_path",
        target_file="some_file",
        allow_no_env=True,
    )
    r.run()


@pytest.mark.usefixtures("use_tmpdir")
def test_run(monkeypatch, source_root):
    with open("rms_config.yml", "w", encoding="utf-8") as f:
        f.write(f"executable:  {os.getcwd()}/bin/rms")

    os.mkdir("run_path")
    os.mkdir("bin")
    os.mkdir("project")
    shutil.copy(os.path.join(source_root, "tests/unit_tests/shared/share/rms"), "bin")
    monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")

    action = {"exit_status": 0}
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    rms_run(0, "project", "workflow", run_path="run_path", allow_no_env=True)

    action = {"exit_status": 1}
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    with pytest.raises(rms.RMSRunException):
        rms_run(0, "project", "workflow", run_path="run_path", allow_no_env=True)

    action = {"exit_status": 0}
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    with pytest.raises(rms.RMSRunException):
        rms_run(
            0,
            "project",
            "workflow",
            run_path="run_path",
            target_file="some_file",
            allow_no_env=True,
        )

    action = {
        "exit_status": 0,
        "target_file": os.path.join(os.getcwd(), "some_file"),
    }

    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))
    rms_run(
        0,
        "project",
        "workflow",
        run_path="run_path",
        target_file="some_file",
        allow_no_env=True,
    )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "val, carry_over",
    [
        ("    ", False),
        ("", False),
        (None, False),
        ("SOME_VAL", True),
    ],
)
def test_rms_load_env(monkeypatch, source_root, val, carry_over):
    # Setup RMS project
    with open("rms_config.yml", "w", encoding="utf-8") as f:
        json.dump(
            {
                "executable": os.path.realpath("bin/rms"),
            },
            f,
        )

    with open("rms_exec_env.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "RMS_TEST_VAR": val,
            },
            f,
        )

    os.mkdir("run_path")
    os.mkdir("bin")
    os.mkdir("project")
    shutil.copy(
        os.path.join(source_root, "tests/unit_tests/shared/share/rms"),
        "bin",
    )
    monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")

    action = {"exit_status": 0}
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    rms_exec = _get_ert_shared_dir() + "/share/ert/forward-models/res/script/rms.py"
    subprocess.check_call(
        [
            rms_exec,
            "--run-path",
            "run_path",
            "0",
            "--version",
            "10.4",
            "project",
            "--import-path",
            "./",
            "--export-path",
            "./",
            "workflow",
            "-a",
        ]
    )

    with open("run_path/env.json", encoding="utf-8") as f:
        env = json.load(f)

    if carry_over:
        assert "RMS_TEST_VAR" in env
    else:
        assert "RMS_TEST_VAR" not in env


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "val, carry_over",
    [
        ("    ", False),
        ("", False),
        (None, False),
        ("SOME_VAL", True),
    ],
)
def test_rms_drop_env(monkeypatch, source_root, val, carry_over):
    # Setup RMS project
    with open("rms_config.yml", "w", encoding="utf-8") as f:
        json.dump(
            {
                "executable": os.path.realpath("bin/rms"),
            },
            f,
        )

    with open("rms_exec_env.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "RMS_TEST_VAR": val,
            },
            f,
        )

    os.mkdir("run_path")
    os.mkdir("bin")
    os.mkdir("project")
    shutil.copy(
        os.path.join(source_root, "tests/unit_tests/shared/share/rms"),
        "bin",
    )
    monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")

    action = {"exit_status": 0}
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    rms_exec = _get_ert_shared_dir() + "/share/ert/forward-models/res/script/rms.py"
    subprocess.check_call(
        [
            rms_exec,
            "--run-path",
            "run_path",
            "0",
            "--version",
            "10.4",
            "project",
            "--import-path",
            "./",
            "--export-path",
            "./",
            "workflow",
            "-a",
        ]
    )

    with open("run_path/env.json", encoding="utf-8") as f:
        env = json.load(f)

    if carry_over:
        assert "RMS_TEST_VAR" in env
    else:
        assert "RMS_TEST_VAR" not in env


@pytest.mark.usefixtures("use_tmpdir")
def test_run_class_with_existing_target_file(monkeypatch, source_root):
    with open("rms_config.yml", "w", encoding="utf-8") as f:
        f.write(f"executable:  {os.getcwd()}/bin/rms")

    os.mkdir("run_path")
    os.mkdir("bin")
    os.mkdir("project")
    shutil.copy(os.path.join(source_root, "tests/unit_tests/shared/share/rms"), "bin")
    monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")

    target_file = os.path.join(os.getcwd(), "rms_target_file")
    action = {
        "exit_status": 0,
        "target_file": target_file,
    }
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    with open(target_file, "w", encoding="utf-8") as f:
        f.write("This is a dummy target file")

    r = rms.RMSRun(
        0,
        "project",
        "workflow",
        run_path="run_path",
        target_file=target_file,
        allow_no_env=True,
    )
    r.run()


@pytest.mark.usefixtures("use_tmpdir")
def test_run_wrapper(monkeypatch, source_root):
    wrapper_file_name = f"{os.getcwd()}/bin/rms_wrapper"
    with open("rms_config.yml", "w", encoding="utf-8") as f:
        f.write(f"executable:  {os.getcwd()}/bin/rms\n")
        f.write(f"wrapper:  {wrapper_file_name}")

    os.mkdir("run_path")
    os.mkdir("bin")
    os.mkdir("project")
    shutil.copy(os.path.join(source_root, "tests/unit_tests/shared/share/rms"), "bin")

    with open(wrapper_file_name, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("exec ${@:1}")
    st = os.stat(wrapper_file_name)
    os.chmod(wrapper_file_name, st.st_mode | stat.S_IEXEC)
    monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")
    monkeypatch.setenv("PATH", f"{os.getcwd()}/bin:{os.environ['PATH']}")

    action = {"exit_status": 0}
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    rms_run(0, "project", "workflow", run_path="run_path", allow_no_env=True)

    action = {"exit_status": 1}
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    with pytest.raises(rms.RMSRunException):
        rms_run(0, "project", "workflow", run_path="run_path", allow_no_env=True)

    action = {"exit_status": 0}
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    with pytest.raises(rms.RMSRunException):
        rms_run(
            0,
            "project",
            "workflow",
            run_path="run_path",
            target_file="some_file",
            allow_no_env=True,
        )

    action = {
        "exit_status": 0,
        "target_file": os.path.join(os.getcwd(), "some_file"),
    }

    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))
    rms_run(
        0,
        "project",
        "workflow",
        run_path="run_path",
        target_file="some_file",
        allow_no_env=True,
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_run_version_env(monkeypatch, source_root):
    wrapper_file_name = f"{os.getcwd()}/bin/rms_wrapper"
    with open("rms_config.yml", "w", encoding="utf-8") as f:
        f.write(
            f"""\
executable: {os.getcwd()}/bin/rms
wrapper:  {wrapper_file_name}
env:
    10.1.3:
        PATH_PREFIX: /some/path
        PYTHONPATH: /some/pythonpath
"""
        )

    os.mkdir("run_path")
    os.mkdir("bin")
    os.mkdir("project")
    shutil.copy(os.path.join(source_root, "tests/unit_tests/shared/share/rms"), "bin")

    with open(wrapper_file_name, "w", encoding="utf-8") as f:
        f.write(
            TEST_ENV_WRAPPER.format(
                expected_path_prefix="/some/path",
                expected_pythonpath="/some/other/pythonpath",
            )
        )

    st = os.stat(wrapper_file_name)
    os.chmod(wrapper_file_name, st.st_mode | stat.S_IEXEC)
    monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")
    monkeypatch.setenv("PATH", f"{os.getcwd()}/bin:{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", "/some/other/pythonpath")

    action = {
        "exit_status": 0,
        "target_file": os.path.join(os.getcwd(), "some_file"),
    }

    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))
    rms_run(
        0,
        "project",
        "workflow",
        run_path="run_path",
        target_file="some_file",
        version="10.1.3",
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_run_version_env_with_user_env(monkeypatch, source_root):
    wrapper_file_name = f"{os.getcwd()}/bin/rms_wrapper"
    with open("rms_config.yml", "w", encoding="utf-8") as f:
        f.write(
            f"""\
executable: {os.getcwd()}/bin/rms
wrapper:  {wrapper_file_name}
env:
    10.1.3:
        PATH_PREFIX: /some/path
        PYTHONPATH: /some/pythonpath
"""
        )

    os.mkdir("run_path")
    os.mkdir("bin")
    os.mkdir("project")
    shutil.copy(os.path.join(source_root, "tests/unit_tests/shared/share/rms"), "bin")

    with open(wrapper_file_name, "w", encoding="utf-8") as f:
        f.write(
            TEST_ENV_WRAPPER.format(
                expected_path_prefix="/some/other/path:/some/path",
                expected_pythonpath="/some/other/pythonpath",
            )
        )
    with open("rms_exec_env.json", "w", encoding="utf-8") as f:
        f.write(
            """\
{
"PATH_PREFIX" : "/some/other/path",
"PYTHONPATH" : "/some/other/pythonpath"
}
"""
        )

    st = os.stat(wrapper_file_name)
    os.chmod(wrapper_file_name, st.st_mode | stat.S_IEXEC)
    monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")
    monkeypatch.setenv("PATH", f"{os.getcwd()}/bin:{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", "/some/other/pythonpath")
    with patch.object(sys, "argv", ["rms"]):
        action = {
            "exit_status": 0,
            "target_file": os.path.join(os.getcwd(), "some_file"),
        }

        with open("run_path/action.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(action))
        rms_run(
            0,
            "project",
            "workflow",
            run_path="run_path",
            target_file="some_file",
            version="10.1.3",
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_run_allow_no_env(monkeypatch, source_root):
    with open("rms_config.yml", "w", encoding="utf-8") as f:
        f.write(
            f"""\
executable: {os.getcwd()}/bin/rms
env:
    10.1.3:
        PATH_PREFIX: /some/path
        PYTHONPATH: /some/pythonpath
"""
        )

    os.mkdir("run_path")
    os.mkdir("bin")
    os.mkdir("project")
    shutil.copy(os.path.join(source_root, "tests/unit_tests/shared/share/rms"), "bin")

    monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")
    monkeypatch.setenv("PATH", f"{os.getcwd()}/bin:{os.environ['PATH']}")
    action = {
        "exit_status": 0,
        "target_file": os.path.join(os.getcwd(), "some_file"),
    }

    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    with pytest.raises(rms.RMSRunException) as e:
        rms_run(
            0,
            "project",
            "workflow",
            run_path="run_path",
            target_file="some_file",
            version="non-existing",
        )
        assert "non-existing" in str(e)

    rms_run(
        0,
        "project",
        "workflow",
        run_path="run_path",
        target_file="some_file",
        version="non-existing",
        allow_no_env=True,
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_rms_job_script_parser(monkeypatch, source_root):
    # Setup RMS project
    with open("rms_config.yml", "w", encoding="utf-8") as f:
        json.dump(
            {
                "executable": os.path.realpath("bin/rms"),
                "env": {"10.1.3": {"PATH": ""}},
            },
            f,
        )

    monkeypatch.setenv("RMS_TEST_VAR", "fdsgfdgfdsgfds")

    os.mkdir("run_path")
    os.mkdir("bin")
    os.mkdir("project")
    shutil.copy(os.path.join(source_root, "tests/unit_tests/shared/share/rms"), "bin")
    monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")

    action = {"exit_status": 0}
    with open("run_path/action.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(action))

    rms_exec = _get_ert_shared_dir() + "/share/ert/forward-models/res/script/rms.py"
    subprocess.check_call(
        [
            rms_exec,
            "--run-path",
            "run_path",
            "0",
            "--version",
            "10.1.3",
            "project",
            "--import-path",
            "./",
            "--export-path",
            "./",
            "workflow",
            "",
        ]
    )

    subprocess.check_call(
        [
            rms_exec,
            "--run-path",
            "run_path",
            "0",
            "--version",
            "10.1.3",
            "project",
            "workflow",
            "-a",
        ]
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_load(monkeypatch):
    monkeypatch.setenv("RMS_SITE_CONFIG", "file/does/not/exist")
    with pytest.raises(IOError):
        conf = rms.RMSConfig()

    monkeypatch.setenv("RMS_SITE_CONFIG", rms.RMSConfig.DEFAULT_CONFIG_FILE)
    conf = rms.RMSConfig()

    with pytest.raises(OSError):
        _ = conf.executable

    with open("file.yml", "w", encoding="utf-8") as f:
        f.write("this:\n -should\n-be\ninvalid:yaml?")

    monkeypatch.setenv("RMS_SITE_CONFIG", "file.yml")
    with pytest.raises(ValueError):
        conf = rms.RMSConfig()

    os.mkdir("bin")
    with open("bin/rms", "w", encoding="utf-8") as f:
        f.write("This is an RMS executable ...")
    os.chmod("bin/rms", stat.S_IEXEC)

    with open("file.yml", "w", encoding="utf-8") as f:
        f.write("executable: bin/rms")

    conf = rms.RMSConfig()
    assert conf.executable == "bin/rms"
    assert conf.threads is None

    with open("file.yml", "w", encoding="utf-8") as f:
        f.write("executable: bin/rms\n")
        f.write("threads: 17")

    conf = rms.RMSConfig()
    assert conf.threads == 17

    with open("file.yml", "w", encoding="utf-8") as f:
        f.write("executable: bin/rms\n")
        f.write("wrapper: not-exisiting-exec")

    conf = rms.RMSConfig()

    with pytest.raises(OSError):
        _ = conf.wrapper

    with open("file.yml", "w", encoding="utf-8") as f:
        f.write("executable: bin/rms\n")
        f.write("wrapper: bash")

    conf = rms.RMSConfig()
    assert conf.wrapper == "bash"


@pytest.mark.usefixtures("use_tmpdir")
def test_load_env(monkeypatch):
    monkeypatch.setenv("RMS_SITE_CONFIG", "file.yml")
    with open("file.yml", "w", encoding="utf-8") as f:
        f.write(
            """\
executable: bin/rms\n
wrapper: bash
env:
    10.1.3:
        PATH_PREFIX: /some/path
        PYTHONPATH: /some/pythonpath
"""
        )
    conf = rms.RMSConfig()
    assert conf.env("10.1.3")["PATH_PREFIX"] == "/some/path"
    assert conf.env("10.1.3")["PYTHONPATH"] == "/some/pythonpath"
    assert conf.env("non_existing") == {}
