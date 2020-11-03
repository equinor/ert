import json
import os
import signal
import stat
import sys
import time
import unittest
import importlib
from subprocess import Popen
from textwrap import dedent

import psutil

from subprocess import Popen
from job_runner.cli import main
from job_runner.reporting.message import Finish
from tests.utils import tmpdir, wait_until
from unittest.mock import patch


class JobDispatchTest(unittest.TestCase):
    @tmpdir(None)
    def test_terminate_jobs(self):

        # Executes it self recursively and sleeps for 100 seconds
        with open("dummy_executable", "w") as f:
            f.write(
                """#!/usr/bin/env python
import sys, os, time
counter = eval(sys.argv[1])
if counter > 0:
    os.fork()
    os.execv(sys.argv[0],[sys.argv[0], str(counter - 1) ])
else:
    time.sleep(100)"""
            )

        executable = os.path.realpath("dummy_executable")
        os.chmod("dummy_executable", stat.S_IRWXU | stat.S_IRWXO | stat.S_IRWXG)

        self.job_list = {
            "umask": "0002",
            "DATA_ROOT": "",
            "global_environment": {},
            "global_update_path": {},
            "jobList": [
                {
                    "name": "dummy_executable",
                    "executable": executable,
                    "target_file": None,
                    "error_file": None,
                    "start_file": None,
                    "stdout": "dummy.stdout",
                    "stderr": "dummy.stderr",
                    "stdin": None,
                    "argList": ["3"],
                    "environment": None,
                    "exec_env": None,
                    "license_path": None,
                    "max_running_minutes": None,
                    "max_running": None,
                    "min_arg": 1,
                    "arg_types": [],
                    "max_arg": None,
                }
            ],
            "run_id": "",
            "ert_pid": "",
        }

        with open("jobs.json", "w") as f:
            f.write(json.dumps(self.job_list))

        # macOS doesn't provide /usr/bin/setsid, so we roll our own
        with open("setsid", "w") as f:
            f.write(
                dedent(
                    """\
                #!/usr/bin/env python
                import os
                import sys
                os.setsid()
                os.execvp(sys.argv[1], sys.argv[1:])
                """
                )
            )
        os.chmod("setsid", 0o755)

        job_dispatch_script = importlib.util.find_spec("job_runner.job_dispatch").origin
        job_dispatch_process = Popen(
            [
                os.getcwd() + "/setsid",
                sys.executable,
                job_dispatch_script,
                os.getcwd(),
            ]
        )

        p = psutil.Process(job_dispatch_process.pid)

        # Three levels of processes should spawn 8 children in total
        wait_until(lambda: self.assertEqual(len(p.children(recursive=True)), 8))

        p.terminate()

        wait_until(lambda: self.assertEqual(len(p.children(recursive=True)), 0))

        os.wait()  # allow os to clean up zombie processes

    @tmpdir(None)
    def test_job_dispatch_kills_itself_after_unsuccessful_job(self):
        with patch("job_runner.cli.os") as mock_os, patch(
            "job_runner.cli.JobRunner"
        ) as mock_runner:
            mock_runner.return_value.run.return_value = [
                Finish().with_error("overall bad run")
            ]
            mock_os.getpgid.return_value = 17

            main(["script.py", "/foo/bar/baz"])

            mock_os.killpg.assert_called_with(17, signal.SIGKILL)

    @tmpdir(None)
    def test_job_dispatch_run_subset_specified_as_parmeter(self):
        with open("dummy_executable", "w") as f:
            f.write(
                "#!/usr/bin/env python\n"
                "import sys, os\n"
                'filename = "job_{}.out".format(sys.argv[1])\n'
                'f = open(filename, "w")\n'
                "f.close()\n"
            )

        executable = os.path.realpath("dummy_executable")
        os.chmod("dummy_executable", stat.S_IRWXU | stat.S_IRWXO | stat.S_IRWXG)

        self.job_list = {
            "umask": "0002",
            "DATA_ROOT": "",
            "global_environment": {},
            "global_update_path": {},
            "jobList": [
                {
                    "name": "job_A",
                    "executable": executable,
                    "target_file": None,
                    "error_file": None,
                    "start_file": None,
                    "stdout": "dummy.stdout",
                    "stderr": "dummy.stderr",
                    "stdin": None,
                    "argList": ["A"],
                    "environment": None,
                    "exec_env": None,
                    "license_path": None,
                    "max_running_minutes": None,
                    "max_running": None,
                    "min_arg": 1,
                    "arg_types": [],
                    "max_arg": None,
                },
                {
                    "name": "job_B",
                    "executable": executable,
                    "target_file": None,
                    "error_file": None,
                    "start_file": None,
                    "stdout": "dummy.stdout",
                    "stderr": "dummy.stderr",
                    "stdin": None,
                    "argList": ["B"],
                    "environment": None,
                    "exec_env": None,
                    "license_path": None,
                    "max_running_minutes": None,
                    "max_running": None,
                    "min_arg": 1,
                    "arg_types": [],
                    "max_arg": None,
                },
                {
                    "name": "job_C",
                    "executable": executable,
                    "target_file": None,
                    "error_file": None,
                    "start_file": None,
                    "stdout": "dummy.stdout",
                    "stderr": "dummy.stderr",
                    "stdin": None,
                    "argList": ["C"],
                    "environment": None,
                    "exec_env": None,
                    "license_path": None,
                    "max_running_minutes": None,
                    "max_running": None,
                    "min_arg": 1,
                    "arg_types": [],
                    "max_arg": None,
                },
            ],
            "run_id": "",
            "ert_pid": "",
        }

        with open("jobs.json", "w") as f:
            f.write(json.dumps(self.job_list))

        # macOS doesn't provide /usr/bin/setsid, so we roll our own
        with open("setsid", "w") as f:
            f.write(
                dedent(
                    """\
                #!/usr/bin/env python
                import os
                import sys
                os.setsid()
                os.execvp(sys.argv[1], sys.argv[1:])
                """
                )
            )
        os.chmod("setsid", 0o755)

        job_dispatch_script = importlib.util.find_spec("job_runner.job_dispatch").origin
        job_dispatch_process = Popen(
            [
                os.getcwd() + "/setsid",
                sys.executable,
                job_dispatch_script,
                os.getcwd(),
                "job_B",
                "job_C",
            ]
        )

        job_dispatch_process.wait()

        assert not os.path.isfile("job_A.out")
        assert os.path.isfile("job_B.out")
        assert os.path.isfile("job_C.out")
