import contextlib
import os
import pathlib
import stat
import sys
import textwrap
from dataclasses import dataclass
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from _ert.forward_model_runner.forward_model_step import (
    ForwardModelStep,
    _get_processtree_data,
)
from _ert.forward_model_runner.reporting.message import Exited, Running, Start


@patch("_ert.forward_model_runner.forward_model_step.check_executable")
@patch("_ert.forward_model_runner.forward_model_step.Popen")
@patch("_ert.forward_model_runner.forward_model_step.Process")
@pytest.mark.usefixtures("use_tmpdir")
def test_run_with_process_failing(mock_process, mock_popen, mock_check_executable):
    fmstep = ForwardModelStep({}, 0)
    mock_check_executable.return_value = ""
    type(mock_process.return_value.memory_info.return_value).rss = PropertyMock(
        return_value=10
    )
    mock_process.return_value.wait.return_value = 9

    run = fmstep.run()

    assert isinstance(next(run), Start), "run did not yield Start message"
    assert isinstance(next(run), Running), "run did not yield Running message"
    exited = next(run)
    assert isinstance(exited, Exited), "run did not yield Exited message"
    assert exited.exit_code == 9, "Exited message had unexpected exit code"

    with pytest.raises(StopIteration):
        next(run)


@pytest.mark.flaky(reruns=10)
@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
def test_cpu_seconds_can_detect_multiprocess():
    """Run a fm step that sets of two simultaneous processes that
    each run for 2 second. We should be able to detect the total
    cpu seconds consumed to be roughly 2 seconds.

    The test is flaky in that it tries to gather cpu_seconds data while
    the subprocesses are running. On a loaded CPU this is not very robust,
    but the most important catch is to be able to obtain a cpu_second
    number that is larger than the busy-wait times of the individual
    sub-processes.
    """
    pythonscript = "busy.py"
    with open(pythonscript, "w", encoding="utf-8") as pyscript:
        pyscript.write(
            textwrap.dedent(
                """\
            import time
            now = time.time()
            while time.time() < now + 2:
                pass"""
            )
        )
    scriptname = "saturate_cpus.sh"
    with open(scriptname, "w", encoding="utf-8") as script:
        script.write(
            textwrap.dedent(
                """\
            #!/bin/sh
            python busy.py &
            python busy.py"""
            )
        )
    executable = os.path.realpath(scriptname)
    os.chmod(scriptname, stat.S_IRWXU | stat.S_IRWXO | stat.S_IRWXG)
    fmstep = ForwardModelStep(
        {
            "executable": executable,
        },
        0,
    )
    fmstep.MEMORY_POLL_PERIOD = 0.05
    cpu_seconds = 0.0
    for status in fmstep.run():
        if isinstance(status, Running):
            cpu_seconds = max(cpu_seconds, status.memory_status.cpu_seconds)
    assert 2.5 < cpu_seconds < 4.5


@pytest.mark.integration_test
@pytest.mark.flaky(reruns=5)
@pytest.mark.usefixtures("use_tmpdir")
def test_memory_usage_counts_grandchildren():
    scriptname = "recursive_memory_hog.py"
    blobsize = 1e7
    with open(scriptname, "w", encoding="utf-8") as script:
        script.write(
            textwrap.dedent(
                """\
            #!/usr/bin/env python
            import os
            import sys
            import time

            counter = int(sys.argv[-2])
            blobsize = int(sys.argv[-1])

            # Allocate memory
            _blob = list(range(blobsize))

            if counter > 0:
                parent = os.fork()
                if not parent:
                    os.execv(sys.argv[-3], [sys.argv[-3], str(counter - 1), str(blobsize)])
            time.sleep(3)"""  # Too low sleep will make the test faster but flaky
            )
        )
    executable = os.path.realpath(scriptname)
    os.chmod(scriptname, stat.S_IRWXU | stat.S_IRWXO | stat.S_IRWXG)

    def max_memory_per_subprocess_layer(layers: int) -> int:
        fmstep = ForwardModelStep(
            {
                "executable": executable,
                "argList": [str(layers), str(int(blobsize))],
            },
            0,
        )
        fmstep.MEMORY_POLL_PERIOD = 0.01
        max_seen = 0
        for status in fmstep.run():
            if isinstance(status, Running):
                max_seen = max(max_seen, status.memory_status.max_rss)
        return max_seen

    # size of the list that gets forked. we will use this when
    # comparing the memory used with different amounts of forks done.
    # subtract a little bit (* 0.9) due to natural variance in memory used
    # when running the program.
    memory_per_numbers_list = sys.getsizeof(0) * blobsize * 0.90

    max_seens = [max_memory_per_subprocess_layer(layers) for layers in range(3)]
    assert max_seens[0] + memory_per_numbers_list < max_seens[1]
    assert max_seens[1] + memory_per_numbers_list < max_seens[2]


@dataclass
class CpuTimes:
    """Mocks the response of psutil.Process().cpu_times()"""

    user: float


@dataclass
class MockedProcess:
    """Mocks psutil.Process()"""

    pid: int
    memory_info = MagicMock()

    def cpu_times(self):
        return CpuTimes(user=self.pid / 10.0)

    def children(self, recursive: bool):
        assert recursive
        if self.pid == 123:
            return [MockedProcess(124)]

    def oneshot(self):
        return contextlib.nullcontext()


def test_cpu_seconds_for_process_with_children():
    (_, cpu_seconds, _, _) = _get_processtree_data(MockedProcess(123))
    assert cpu_seconds == 123 / 10.0 + 124 / 10.0


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No oom_score on MacOS")
def test_oom_score_is_max_over_processtree():
    def read_text_side_effect(self: pathlib.Path, *args, **kwargs):
        if self.absolute() == pathlib.Path("/proc/123/oom_score"):
            return "234"
        if self.absolute() == pathlib.Path("/proc/124/oom_score"):
            return "456"

    with patch("pathlib.Path.read_text", autospec=True) as mocked_read_text:
        mocked_read_text.side_effect = read_text_side_effect
        (_, _, oom_score, _) = _get_processtree_data(MockedProcess(123))

    assert oom_score == 456


@pytest.mark.usefixtures("use_tmpdir")
def test_run_fails_using_exit_bash_builtin():
    fmstep = ForwardModelStep(
        {
            "name": "exit 1",
            "executable": "/bin/sh",
            "stdout": "exit_out",
            "stderr": "exit_err",
            "argList": ["-c", 'echo "failed with 1" 1>&2 ; exit 1'],
        },
        0,
    )

    statuses = list(fmstep.run())

    assert len(statuses) == 3, "Wrong statuses count"
    assert statuses[2].exit_code == 1, "Exited status wrong exit_code"
    assert statuses[2].error_message == "Process exited with status code 1", (
        "Exited status wrong error_message"
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_run_with_defined_executable_but_missing():
    executable = os.path.join(os.getcwd(), "this/is/not/a/file")
    fmstep = ForwardModelStep(
        {
            "name": "TEST_EXECUTABLE_NOT_FOUND",
            "executable": executable,
            "stdout": "mkdir_out",
            "stderr": "mkdir_err",
        },
        0,
    )

    start_message = next(fmstep.run())
    assert isinstance(start_message, Start)
    assert "this/is/not/a/file is not a file" in start_message.error_message


@pytest.mark.usefixtures("use_tmpdir")
def test_run_with_empty_executable():
    empty_executable = os.path.join(os.getcwd(), "foo")
    with open(empty_executable, "a", encoding="utf-8"):
        pass
    st = os.stat(empty_executable)
    os.chmod(empty_executable, st.st_mode | stat.S_IEXEC)

    fmstep = ForwardModelStep(
        {
            "name": "TEST_EXECUTABLE_NOT_EXECUTABLE",
            "executable": empty_executable,
            "stdout": "mkdir_out",
            "stderr": "mkdir_err",
        },
        0,
    )
    run_status = list(fmstep.run())
    assert len(run_status) == 2
    start_msg, exit_msg = run_status
    assert isinstance(start_msg, Start)
    assert isinstance(exit_msg, Exited)
    assert exit_msg.exit_code == 8
    assert "Missing execution format information" in exit_msg.error_message


@pytest.mark.usefixtures("use_tmpdir")
def test_run_with_defined_executable_no_exec_bit():
    non_executable = os.path.join(os.getcwd(), "foo")
    with open(non_executable, "a", encoding="utf-8"):
        pass

    fmstep = ForwardModelStep(
        {
            "name": "TEST_EXECUTABLE_NOT_EXECUTABLE",
            "executable": non_executable,
            "stdout": "mkdir_out",
            "stderr": "mkdir_err",
        },
        0,
    )
    start_message = next(fmstep.run())
    assert isinstance(start_message, Start)
    assert "foo is not an executable" in start_message.error_message


def test_init_fmstep_no_std():
    fmstep = ForwardModelStep(
        {},
        0,
    )
    assert fmstep.std_err is None
    assert fmstep.std_out is None


def test_init_fmstep_with_std():
    fmstep = ForwardModelStep(
        {
            "stdout": "exit_out",
            "stderr": "exit_err",
        },
        0,
    )
    assert fmstep.std_err == "exit_err"
    assert fmstep.std_out == "exit_out"


def test_makedirs(monkeypatch, tmp_path):
    """
    Test that the directories for the output process streams are created if
    they don't exist
    """
    monkeypatch.chdir(tmp_path)
    fmstep = ForwardModelStep(
        {
            "executable": "true",
            "stdout": "a/file",
            "stderr": "b/c/file",
        },
        0,
    )
    for _ in fmstep.run():
        pass
    assert (tmp_path / "a/file").is_file()
    assert (tmp_path / "b/c/file").is_file()
