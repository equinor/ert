import contextlib
import os
import pathlib
import stat
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from _ert.forward_model_runner.job import Job, _get_processtree_data
from _ert.forward_model_runner.reporting.message import Exited, Running, Start


@patch("_ert.forward_model_runner.job.assert_file_executable")
@patch("_ert.forward_model_runner.job.Popen")
@patch("_ert.forward_model_runner.job.Process")
@pytest.mark.usefixtures("use_tmpdir")
def test_run_with_process_failing(
    mock_process, mock_popen, mock_assert_file_executable
):
    job = Job({}, 0)
    type(mock_process.return_value.memory_info.return_value).rss = PropertyMock(
        return_value=10
    )
    mock_process.return_value.wait.return_value = 9

    run = job.run()

    assert isinstance(next(run), Start), "run did not yield Start message"
    assert isinstance(next(run), Running), "run did not yield Running message"
    exited = next(run)
    assert isinstance(exited, Exited), "run did not yield Exited message"
    assert exited.exit_code == 9, "Exited message had unexpected exit code"

    with pytest.raises(StopIteration):
        next(run)


@pytest.mark.flaky(reruns=5)
@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
def test_cpu_seconds_can_detect_multiprocess():
    """Run a job that sets of two simultaneous processes that
    each run for 1 second. We should be able to detect the total
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
            while time.time() < now + 1:
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
    job = Job(
        {
            "executable": executable,
        },
        0,
    )
    job.MEMORY_POLL_PERIOD = 0.1
    cpu_seconds = 0.0
    for status in job.run():
        if isinstance(status, Running):
            cpu_seconds = max(cpu_seconds, status.memory_status.cpu_seconds)
    assert 1.4 < cpu_seconds < 2.2


@pytest.mark.integration_test
@pytest.mark.flaky(reruns=5)
@pytest.mark.usefixtures("use_tmpdir")
def test_memory_usage_counts_grandchildren():
    scriptname = "recursive_memory_hog.py"
    with open(scriptname, "w", encoding="utf-8") as script:
        script.write(
            textwrap.dedent(
                """\
            #!/usr/bin/env python
            import os
            import sys
            import time

            counter = int(sys.argv[-2])
            numbers = list(range(int(sys.argv[-1])))
            if counter > 0:
                parent = os.fork()
                if not parent:
                    os.execv(sys.argv[-3], [sys.argv[-3], str(counter - 1), str(int(1e7))])
            time.sleep(1)"""  # Too low sleep will make the test faster but flaky
            )
        )
    executable = os.path.realpath(scriptname)
    os.chmod(scriptname, stat.S_IRWXU | stat.S_IRWXO | stat.S_IRWXG)

    def max_memory_per_subprocess_layer(layers: int) -> int:
        job = Job(
            {
                "executable": executable,
                "argList": [str(layers), str(int(1e6))],
            },
            0,
        )
        job.MEMORY_POLL_PERIOD = 0.01
        max_seen = 0
        for status in job.run():
            if isinstance(status, Running):
                max_seen = max(max_seen, status.memory_status.max_rss)
        return max_seen

    # size of the list that gets forked. we will use this when
    # comparing the memory used with different amounts of forks done.
    # subtract a little bit (* 0.9) due to natural variance in memory used
    # when running the program.
    memory_per_numbers_list = sys.getsizeof(int(0)) * 1e7 * 0.90

    max_seens = [max_memory_per_subprocess_layer(layers) for layers in range(3)]
    assert max_seens[0] + memory_per_numbers_list < max_seens[1]
    assert max_seens[1] + memory_per_numbers_list < max_seens[2]


@pytest.mark.integration_test
@pytest.mark.flaky(reruns=5)
@pytest.mark.usefixtures("use_tmpdir")
def test_memory_profile_in_running_events():
    scriptname = "increasing_memory.py"
    with open(scriptname, "w", encoding="utf-8") as script:
        script.write(
            textwrap.dedent(
                """\
            #!/usr/bin/env python
            import time
            somelist = []

            for _ in range(10):
                # 1 Mb allocated pr iteration
                somelist.append(b' ' * 1024 * 1024)
                time.sleep(0.1)"""
            )
        )
    executable = os.path.realpath(scriptname)
    os.chmod(scriptname, stat.S_IRWXU | stat.S_IRWXO | stat.S_IRWXG)

    fm_step = Job(
        {
            "executable": executable,
            "argList": [""],
        },
        0,
    )
    fm_step.MEMORY_POLL_PERIOD = 0.01
    emitted_timestamps: List[datetime] = []
    emitted_rss_values: List[Optional[int]] = []
    emitted_oom_score_values: List[Optional[int]] = []
    for status in fm_step.run():
        if isinstance(status, Running):
            emitted_timestamps.append(
                datetime.fromisoformat(status.memory_status.timestamp)
            )
            emitted_rss_values.append(status.memory_status.rss)
            emitted_oom_score_values.append(status.memory_status.oom_score)

    # Any asserts on the emitted_rss_values easily becomes flaky, so be mild:
    assert (
        np.diff(np.array(emitted_rss_values[:-3])) >= 0
        # Avoid the tail of the array, then the process is tearing down
    ).all(), f"Emitted memory usage not increasing, got {emitted_rss_values[:-3]=}"

    memory_deltas = np.diff(np.array(emitted_rss_values[7:]))
    if not len(memory_deltas):
        # This can happen if memory profiling is lagging behind the process
        # we are trying to track.
        memory_deltas = np.diff(np.array(emitted_rss_values[2:]))

    lenience_factor = 4
    # Ideally this is 1 which corresponds to being able to track every memory
    # allocation perfectly. But on loaded hardware, some of the allocations can be
    # missed due to process scheduling. Bump as needed.

    assert (
        max(memory_deltas) < lenience_factor * 1024 * 1024
        # Avoid the first steps, which includes the Python interpreters memory usage
    ), (
        "Memory increased too sharply, missing a measurement? "
        f"Got {emitted_rss_values=} with selected diffs {memory_deltas}. "
        "If the maximal number is at the beginning, it is probably the Python process "
        "startup that is tracked."
    )

    if sys.platform.startswith("darwin"):
        # No oom_score on MacOS
        assert set(emitted_oom_score_values) == {None}
    else:
        for oom_score in emitted_oom_score_values:
            assert oom_score is not None, "No oom_score, are you not on Linux?"
            # Upper limit "should" be 1000, but has been proven to overshoot.
            assert oom_score >= -1000

    timedeltas = np.diff(np.array(emitted_timestamps))
    # The timedeltas should be close to MEMORY_POLL_PERIOD==0.01, but
    # any weak test hardware will make that hard to attain.
    assert min(timedeltas).total_seconds() >= 0.01


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
    (_, cpu_seconds, _) = _get_processtree_data(MockedProcess(123))
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
        (_, _, oom_score) = _get_processtree_data(MockedProcess(123))

    assert oom_score == 456


@pytest.mark.usefixtures("use_tmpdir")
def test_run_fails_using_exit_bash_builtin():
    job = Job(
        {
            "name": "exit 1",
            "executable": "/bin/sh",
            "stdout": "exit_out",
            "stderr": "exit_err",
            "argList": ["-c", 'echo "failed with 1" 1>&2 ; exit 1'],
        },
        0,
    )

    statuses = list(job.run())

    assert len(statuses) == 3, "Wrong statuses count"
    assert statuses[2].exit_code == 1, "Exited status wrong exit_code"
    assert (
        statuses[2].error_message == "Process exited with status code 1"
    ), "Exited status wrong error_message"


@pytest.mark.usefixtures("use_tmpdir")
def test_run_with_defined_executable_but_missing():
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

    with pytest.raises(IOError):
        for _ in job.run():
            pass


@pytest.mark.usefixtures("use_tmpdir")
def test_run_with_empty_executable():
    empty_executable = os.path.join(os.getcwd(), "foo")
    with open(empty_executable, "a", encoding="utf-8"):
        pass
    st = os.stat(empty_executable)
    os.chmod(empty_executable, st.st_mode | stat.S_IEXEC)

    job = Job(
        {
            "name": "TEST_EXECUTABLE_NOT_EXECUTABLE",
            "executable": empty_executable,
            "stdout": "mkdir_out",
            "stderr": "mkdir_err",
        },
        0,
    )
    run_status = list(job.run())
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

    job = Job(
        {
            "name": "TEST_EXECUTABLE_NOT_EXECUTABLE",
            "executable": non_executable,
            "stdout": "mkdir_out",
            "stderr": "mkdir_err",
        },
        0,
    )

    with pytest.raises(IOError):
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
            "executable": "true",
            "stdout": "a/file",
            "stderr": "b/c/file",
        },
        0,
    )
    for _ in job.run():
        pass
    assert (tmp_path / "a/file").is_file()
    assert (tmp_path / "b/c/file").is_file()
