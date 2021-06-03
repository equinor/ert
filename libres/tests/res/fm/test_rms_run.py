#  Copyright (C) 2018  Equinor ASA, Norway.
#
#  The file 'test_rms.py' is part of ERT - Ensemble based Reservoir Tool.
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
import sys
import stat
import subprocess
import unittest
from unittest.mock import patch
import yaml
import shutil
import json
from ecl.util.test import TestAreaContext
from tests import ResTest
from pytest import MonkeyPatch

from res.fm.rms import RMSRun, RMSRunException
import res.fm.rms

from tests.utils import tmpdir
import pytest
from tests.conftest import source_root


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
if [[ $PYTHONPATH != $PYPATH_EXPECTED ]] # first user defined, then config defined, then rest
then
    echo "PYTHONPATH set incorrectly"
    echo $PYTHONPATH should be $PYPATH_EXPECTED
    exit 1
fi
$@
"""


def _mocked_run(**kwargs):
    print(kwargs)


@tmpdir()
@pytest.mark.parametrize(
    "test_input,expected_result",
    [
        (0, 422851785),
        (1, 723121249),
        (2, 132312123),
    ],
)
def test_run_class_multi_seed(tmpdir, monkeypatch, test_input, expected_result):
    with open("rms_config.yml", "w") as f:
        f.write("executable:  {}/bin/rms".format(os.getcwd()))

    os.mkdir("test_run_multi_seed")
    os.mkdir("run_path")
    os.mkdir("bin")
    os.mkdir("project")
    shutil.copy(os.path.join(source_root(), "tests/res/fm/rms"), "bin")
    monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")

    action = {"exit_status": 0}
    with open("run_path/action.json", "w") as f:
        f.write(json.dumps(action))

    seed_file = ["3", "422851785", "723121249", "132312123"]
    with open("run_path/random.seeds", "w") as f:
        f.write("\n".join(seed_file))

    r = RMSRun(test_input, "project", "workflow", run_path="run_path")
    assert r.seed == expected_result


class RMSRunTest(ResTest):
    def setUp(self):
        self.monkeypatch = MonkeyPatch()
        pass

    def tearDown(self):
        self.monkeypatch.undo()

    def test_create(self):
        with self.assertRaises(OSError):
            r = RMSRun(0, "/project/does/not/exist", "workflow")

        with TestAreaContext("test_create"):
            os.mkdir("rms")
            r = RMSRun(0, "rms", "workflow")

    def test_run_class(self):
        with TestAreaContext("test_run"):
            with open("rms_config.yml", "w") as f:
                f.write("executable:  {}/bin/rms".format(os.getcwd()))

            os.mkdir("run_path")
            os.mkdir("bin")
            os.mkdir("project")
            shutil.copy(os.path.join(self.SOURCE_ROOT, "tests/res/fm/rms"), "bin")
            self.monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")

            action = {"exit_status": 0}
            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))

            r = RMSRun(0, "project", "workflow", run_path="run_path", allow_no_env=True)
            r.run()

            # -----------------------------------------------------------------

            action = {"exit_status": 1}
            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))

            r = RMSRun(0, "project", "workflow", run_path="run_path", allow_no_env=True)
            with self.assertRaises(RMSRunException):
                r.run()

            # -----------------------------------------------------------------

            action = {"exit_status": 0}
            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))

            r = RMSRun(
                0,
                "project",
                "workflow",
                run_path="run_path",
                target_file="some_file",
                allow_no_env=True,
            )
            with self.assertRaises(RMSRunException):
                r.run()

            # -----------------------------------------------------------------

            action = {
                "exit_status": 0,
                "target_file": os.path.join(os.getcwd(), "some_file"),
            }
            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))

            r = RMSRun(
                0,
                "project",
                "workflow",
                run_path="run_path",
                target_file="some_file",
                allow_no_env=True,
            )
            r.run()

    def test_run(self):
        with TestAreaContext("test_run"):
            with open("rms_config.yml", "w") as f:
                f.write("executable:  {}/bin/rms".format(os.getcwd()))

            os.mkdir("run_path")
            os.mkdir("bin")
            os.mkdir("project")
            shutil.copy(os.path.join(self.SOURCE_ROOT, "tests/res/fm/rms"), "bin")
            self.monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")

            action = {"exit_status": 0}
            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))

            res.fm.rms.run(
                0, "project", "workflow", run_path="run_path", allow_no_env=True
            )

            # -----------------------------------------------------------------

            action = {"exit_status": 1}
            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))

            with self.assertRaises(RMSRunException):
                res.fm.rms.run(
                    0, "project", "workflow", run_path="run_path", allow_no_env=True
                )

            # -----------------------------------------------------------------

            action = {"exit_status": 0}
            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))

            with self.assertRaises(RMSRunException):
                res.fm.rms.run(
                    0,
                    "project",
                    "workflow",
                    run_path="run_path",
                    target_file="some_file",
                    allow_no_env=True,
                )

            # -----------------------------------------------------------------

            action = {
                "exit_status": 0,
                "target_file": os.path.join(os.getcwd(), "some_file"),
            }

            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))
            res.fm.rms.run(
                0,
                "project",
                "workflow",
                run_path="run_path",
                target_file="some_file",
                allow_no_env=True,
            )

    def test_rms_load_env(self):
        test_bed = [
            ("    ", False),
            ("", False),
            (None, False),
            ("SOME_VAL", True),
        ]
        for val, carry_over in test_bed:
            with TestAreaContext("test_drop_path"):
                # Setup RMS project
                with open("rms_config.yml", "w") as f:
                    json.dump(
                        {
                            "executable": os.path.realpath("bin/rms"),
                        },
                        f,
                    )

                with open("rms_exec_env.json", "w") as f:
                    json.dump(
                        {
                            "RMS_TEST_VAR": val,
                        },
                        f,
                    )

                os.mkdir("run_path")
                os.mkdir("bin")
                os.mkdir("project")
                shutil.copy(os.path.join(self.SOURCE_ROOT, "tests/res/fm/rms"), "bin")
                self.monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")

                action = {"exit_status": 0}
                with open("run_path/action.json", "w") as f:
                    f.write(json.dumps(action))

                rms_exec = os.path.join(
                    self.SOURCE_ROOT, "share/ert/forward-models/res/script/rms"
                )
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

                with open("run_path/env.json") as f:
                    env = json.load(f)

                if carry_over:
                    self.assertIn("RMS_TEST_VAR", env)
                else:
                    self.assertNotIn("RMS_TEST_VAR", env)

    def test_rms_drop_env(self):
        test_bed = [
            ("    ", False),
            ("", False),
            (None, False),
            ("SOME_VAL", True),
        ]
        for val, carry_over in test_bed:
            with TestAreaContext("test_drop_path"):
                # Setup RMS project
                with open("rms_config.yml", "w") as f:
                    json.dump(
                        {
                            "executable": os.path.realpath("bin/rms"),
                        },
                        f,
                    )

                with open("rms_exec_env.json", "w") as f:
                    json.dump(
                        {
                            "RMS_TEST_VAR": val,
                        },
                        f,
                    )
                self.monkeypatch.setenv("RMS_TEST_VAR", "fdsgfdgfdsgfds")

                os.mkdir("run_path")
                os.mkdir("bin")
                os.mkdir("project")
                shutil.copy(os.path.join(self.SOURCE_ROOT, "tests/res/fm/rms"), "bin")
                self.monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")

                action = {"exit_status": 0}
                with open("run_path/action.json", "w") as f:
                    f.write(json.dumps(action))

                rms_exec = os.path.join(
                    self.SOURCE_ROOT, "share/ert/forward-models/res/script/rms"
                )
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

                with open("run_path/env.json") as f:
                    env = json.load(f)

                if carry_over:
                    self.assertIn("RMS_TEST_VAR", env)
                else:
                    self.assertNotIn("RMS_TEST_VAR", env)

    def test_run_class_with_existing_target_file(self):
        with TestAreaContext("test_run_existing_target"):
            with open("rms_config.yml", "w") as f:
                f.write("executable:  {}/bin/rms".format(os.getcwd()))

            os.mkdir("run_path")
            os.mkdir("bin")
            os.mkdir("project")
            shutil.copy(os.path.join(self.SOURCE_ROOT, "tests/res/fm/rms"), "bin")
            self.monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")

            target_file = os.path.join(os.getcwd(), "rms_target_file")
            action = {
                "exit_status": 0,
                "target_file": target_file,
            }
            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))

            with open(target_file, "w") as f:
                f.write("This is a dummy target file")

            r = RMSRun(
                0,
                "project",
                "workflow",
                run_path="run_path",
                target_file=target_file,
                allow_no_env=True,
            )
            r.run()

    def test_run_wrapper(self):
        with TestAreaContext("test_run"):
            wrapper_file_name = f"{os.getcwd()}/bin/rms_wrapper"
            with open("rms_config.yml", "w") as f:
                f.write("executable:  {}/bin/rms\n".format(os.getcwd()))
                f.write(f"wrapper:  {wrapper_file_name}")

            os.mkdir("run_path")
            os.mkdir("bin")
            os.mkdir("project")
            shutil.copy(os.path.join(self.SOURCE_ROOT, "tests/res/fm/rms"), "bin")

            with open(wrapper_file_name, "w") as f:
                f.write("#!/bin/bash\n")
                f.write("exec ${@:1}")
            st = os.stat(wrapper_file_name)
            os.chmod(wrapper_file_name, st.st_mode | stat.S_IEXEC)
            self.monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")
            self.monkeypatch.setenv("PATH", f"{os.getcwd()}/bin:{os.environ['PATH']}")

            action = {"exit_status": 0}
            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))

            res.fm.rms.run(
                0, "project", "workflow", run_path="run_path", allow_no_env=True
            )

            # -----------------------------------------------------------------

            action = {"exit_status": 1}
            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))

            with self.assertRaises(RMSRunException):
                res.fm.rms.run(
                    0, "project", "workflow", run_path="run_path", allow_no_env=True
                )

            # -----------------------------------------------------------------

            action = {"exit_status": 0}
            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))

            with self.assertRaises(RMSRunException):
                res.fm.rms.run(
                    0,
                    "project",
                    "workflow",
                    run_path="run_path",
                    target_file="some_file",
                    allow_no_env=True,
                )

            # -----------------------------------------------------------------

            action = {
                "exit_status": 0,
                "target_file": os.path.join(os.getcwd(), "some_file"),
            }

            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))
            res.fm.rms.run(
                0,
                "project",
                "workflow",
                run_path="run_path",
                target_file="some_file",
                allow_no_env=True,
            )

    def test_run_version_env(self):
        with TestAreaContext("test_run"):
            wrapper_file_name = f"{os.getcwd()}/bin/rms_wrapper"
            with open("rms_config.yml", "w") as f:
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
            shutil.copy(os.path.join(self.SOURCE_ROOT, "tests/res/fm/rms"), "bin")

            with open(wrapper_file_name, "w") as f:
                f.write(
                    TEST_ENV_WRAPPER.format(
                        expected_path_prefix="/some/path",
                        expected_pythonpath="/some/pythonpath",
                    )
                )

            st = os.stat(wrapper_file_name)
            os.chmod(wrapper_file_name, st.st_mode | stat.S_IEXEC)
            self.monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")
            self.monkeypatch.setenv("PATH", f"{os.getcwd()}/bin:{os.environ['PATH']}")

            action = {
                "exit_status": 0,
                "target_file": os.path.join(os.getcwd(), "some_file"),
            }

            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))
            res.fm.rms.run(
                0,
                "project",
                "workflow",
                run_path="run_path",
                target_file="some_file",
                version="10.1.3",
            )

    def test_run_version_env_with_user_env(self):
        with TestAreaContext("test_run"):
            wrapper_file_name = f"{os.getcwd()}/bin/rms_wrapper"
            with open("rms_config.yml", "w") as f:
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
            shutil.copy(os.path.join(self.SOURCE_ROOT, "tests/res/fm/rms"), "bin")

            with open(wrapper_file_name, "w") as f:
                f.write(
                    TEST_ENV_WRAPPER.format(
                        expected_path_prefix="/some/other/path:/some/path",
                        expected_pythonpath="/some/other/pythonpath:/some/pythonpath",
                    )
                )
            with open("rms_exec_env.json", "w") as f:
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
            self.monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")
            self.monkeypatch.setenv("PATH", f"{os.getcwd()}/bin:{os.environ['PATH']}")

            with patch.object(sys, "argv", ["rms"]):
                action = {
                    "exit_status": 0,
                    "target_file": os.path.join(os.getcwd(), "some_file"),
                }

                with open("run_path/action.json", "w") as f:
                    f.write(json.dumps(action))
                res.fm.rms.run(
                    0,
                    "project",
                    "workflow",
                    run_path="run_path",
                    target_file="some_file",
                    version="10.1.3",
                )

    def test_run_allow_no_env(self):
        with TestAreaContext("test_run"):
            wrapper_file_name = f"{os.getcwd()}/bin/rms_wrapper"
            with open("rms_config.yml", "w") as f:
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
            shutil.copy(os.path.join(self.SOURCE_ROOT, "tests/res/fm/rms"), "bin")

            self.monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")
            self.monkeypatch.setenv("PATH", f"{os.getcwd()}/bin:{os.environ['PATH']}")
            action = {
                "exit_status": 0,
                "target_file": os.path.join(os.getcwd(), "some_file"),
            }

            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))

            with self.assertRaises(RMSRunException) as e:
                res.fm.rms.run(
                    0,
                    "project",
                    "workflow",
                    run_path="run_path",
                    target_file="some_file",
                    version="non-existing",
                )
                assert "non-existing" in str(e)

            res.fm.rms.run(
                0,
                "project",
                "workflow",
                run_path="run_path",
                target_file="some_file",
                version="non-existing",
                allow_no_env=True,
            )

    def test_rms_job_script_parser(self):
        with TestAreaContext("test_run"):
            # Setup RMS project
            with open("rms_config.yml", "w") as f:
                json.dump(
                    {
                        "executable": os.path.realpath("bin/rms"),
                        "env": {"10.1.3": {"PATH": ""}},
                    },
                    f,
                )

            self.monkeypatch.setenv("RMS_TEST_VAR", "fdsgfdgfdsgfds")

            os.mkdir("run_path")
            os.mkdir("bin")
            os.mkdir("project")
            shutil.copy(os.path.join(self.SOURCE_ROOT, "tests/res/fm/rms"), "bin")
            self.monkeypatch.setenv("RMS_SITE_CONFIG", "rms_config.yml")

            action = {"exit_status": 0}
            with open("run_path/action.json", "w") as f:
                f.write(json.dumps(action))

            rms_exec = os.path.join(
                self.SOURCE_ROOT, "share/ert/forward-models/res/script/rms"
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


if __name__ == "__main__":
    unittest.main()
