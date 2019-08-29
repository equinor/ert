import json
import os
import os.path
from unittest import TestCase

from job_runner.reporting.message import Exited, Finish, Start
from job_runner.runner import JobRunner

from tests.utils import tmpdir


class JobManagerTestRuntimeKW(TestCase):
    @tmpdir(None)
    def test_run_one_job_with_an_integer_arg_is_actually_a_fractional(self):
        executable = "echo"

        job_0 = {"name": "JOB_1",
                 "executable": executable,
                 "stdout": "outfile.stdout.1",
                 "stderr": None,
                 "argList": ['a_file', '5.12'],
                 "min_arg": 1,
                 "max_arg": 2,
                 "arg_types": ['STRING', 'RUNTIME_INT']}

        data = {"umask": "0000",
                "DATA_ROOT": "/path/to/data",
                "jobList": [job_0]}

        jobs_file = os.path.join(os.getcwd(), "jobs.json")
        with open(jobs_file, "w") as f:
            f.write(json.dumps(data))
        runner = JobRunner()
        statuses = list(runner.run([]))
        starts = [e for e in statuses if isinstance(e, Start)]

        self.assertEqual(1, len(starts), "There should be 1 start message")
        self.assertFalse(starts[0].success(),
                         "job should not start with success")

    @tmpdir(None)
    def test_run_given_one_job_with_missing_file_and_one_file_present(self):
        with open("a_file", "w") as f:
            f.write("Hello")

        executable = "echo"

        job_1 = {"name": "JOB_0",
                 "executable": executable,
                 "stdout": "outfile.stdout.0",
                 "stderr": None,
                 "argList": ['some_file'],
                 "min_arg": 1,
                 "max_arg": 1,
                 "arg_types": ['RUNTIME_FILE']}

        job_0 = {"name": "JOB_1",
                 "executable": executable,
                 "stdout": "outfile.stdout.1",
                 "stderr": None,
                 "argList": ['5', 'a_file'],
                 "min_arg": 1,
                 "max_arg": 2,
                 "arg_types": ['RUNTIME_INT', 'RUNTIME_FILE']}

        data = {"umask": "0000",
                "DATA_ROOT": "/path/to/data",
                "jobList": [job_0, job_1]}

        jobs_file = os.path.join(os.getcwd(), "jobs.json")
        with open(jobs_file, "w") as f:
            f.write(json.dumps(data))

        runner = JobRunner()

        statuses = list(runner.run([]))

        starts = [e for e in statuses if isinstance(e, Start)]
        self.assertEqual(2, len(starts), "There should be 2 start messages")
        self.assertTrue(starts[0].success(),
                        "first job should start with success")
        self.assertFalse(starts[1].success(),
                         "second job should not start with success")

        exits = [e for e in statuses if isinstance(e, Exited)]
        self.assertEqual(1, len(exits), "There should be 1 exit message")
        self.assertTrue(exits[0].success(),
                        "first job should exit with success")

        self.assertEqual(Finish, type(
            statuses[-1]), "last message should be Finish")
        self.assertFalse(statuses[-1].success(),
                         "Finish status should not be success")
