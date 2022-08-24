import os
from unittest import TestCase
from unittest.mock import PropertyMock, patch

from ert.job_runner.job import Job
from ert.job_runner.reporting.message import Exited, Running, Start

from ..libres_utils import tmpdir


class JobTests(TestCase):
    @patch("ert.job_runner.job.assert_file_executable")
    @patch("ert.job_runner.job.Popen")
    @patch("ert.job_runner.job.Process")
    @tmpdir(None)
    def test_run_with_process_failing(
        self, mock_process, mock_popen, mock_assert_file_executable
    ):
        job = Job({}, 0)
        type(mock_process.return_value.memory_info.return_value).rss = PropertyMock(
            return_value=10
        )
        mock_process.return_value.wait.return_value = 9

        run = job.run()

        self.assertIsInstance(next(run), Start, "run did not yield Start message")
        self.assertIsInstance(next(run), Running, "run did not yield Running message")
        exited = next(run)
        self.assertIsInstance(exited, Exited, "run did not yield Exited message")
        self.assertEqual(9, exited.exit_code, "Exited message had unexpected exit code")

        with self.assertRaises(StopIteration):
            next(run)

    @tmpdir(None)
    def test_run_fails_using_exit_bash_builtin(self):
        job = Job(
            {
                "name": "exit 1",
                "executable": "/bin/bash",
                "stdout": "exit_out",
                "stderr": "exit_err",
                "argList": ["-c", 'echo "failed with 1" 1>&2 ; exit 1'],
            },
            0,
        )

        statuses = list(job.run())

        self.assertEqual(3, len(statuses), "Wrong statuses count")
        self.assertEqual(1, statuses[2].exit_code, "Exited status wrong exit_code")
        self.assertEqual(
            "Process exited with status code 1",
            statuses[2].error_message,
            "Exited status wrong error_message",
        )

    @tmpdir(None)
    def test_run_with_defined_executable_but_missing(self):
        executable = os.path.join(os.getcwd(), "this/is/not/a/file")
        job = Job(
            {
                "name": "TEST_EXECUTABLE_NOT_FOUND",
                "executable": executable,
                "stdout": "mkdir_out",
                "stderr": "mkdir_err",
            },
            0,
        )

        with self.assertRaises(IOError):
            for _ in job.run():
                pass

    @tmpdir(None)
    def test_run_with_defined_executable_no_exec_bit(self):
        non_executable = os.path.join(os.getcwd(), "foo")
        with open(non_executable, "a"):
            pass

        job = Job(
            {
                "name": "TEST_EXECUTABLE_NOT_EXECUTABLE",
                "executable": non_executable,
                "stdout": "mkdir_out",
                "stderr": "mkdir_err",
            },
            0,
        )

        with self.assertRaises(IOError):
            for _ in job.run():
                pass


def test_init_job_no_std():
    job = Job(
        {},
        0,
    )
    assert job.std_err is None
    assert job.std_out is None


def test_init_job_with_std():
    job = Job(
        {
            "stdout": "exit_out",
            "stderr": "exit_err",
        },
        0,
    )
    assert job.std_err == "exit_err"
    assert job.std_out == "exit_out"


def test_makedirs(monkeypatch, tmp_path):
    """
    Test that the directories for the output process streams are created if
    they don't exist
    """
    monkeypatch.chdir(tmp_path)
    job = Job(
        {
            "executable": "/usr/bin/true",
            "stdout": "a/file",
            "stderr": "b/c/file",
        },
        0,
    )
    for _ in job.run():
        pass
    assert (tmp_path / "a/file").is_file()
    assert (tmp_path / "b/c/file").is_file()
