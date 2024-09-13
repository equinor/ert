from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime as dt
from pathlib import Path
from subprocess import Popen, run
from typing import Optional, Tuple

from psutil import AccessDenied, NoSuchProcess, Process, TimeoutExpired, ZombieProcess

from .io import assert_file_executable
from .reporting.message import (
    Exited,
    ProcessTreeStatus,
    Running,
    Start,
)

logger = logging.getLogger(__name__)


def killed_by_oom(pids: set[int]) -> bool:
    """Will try to detect if a process (or any of its descendants) was killed
    by the Linux OOM-killer.

    Debug information will be logged through the system logger.

    Since pids can be reused, this can in theory give false positives.
    """

    if sys.platform == "darwin":
        return False

    try:
        dmesg_result = run("dmesg", capture_output=True, check=False)
        if dmesg_result.returncode != 0:
            logger.warning(
                "Could not use dmesg to check for OOM kill, "
                f"returncode {dmesg_result.returncode} and stderr: {dmesg_result.stderr}"
            )
            return False
    except FileNotFoundError:
        logger.warning(
            "Could not use dmesg to check for OOM kill, utility not available"
        )
        return False

    oom_lines = "".join(
        [
            line
            for line in dmesg_result.stdout.decode(
                "ascii", errors="ignore"
            ).splitlines()
            if "Out of memory:" in line
        ]
    )

    for pid in pids:
        rhel7_message = f"Kill process {pid}"
        rhel8_message = f"Killed process {pid}"
        if rhel7_message in oom_lines or rhel8_message in oom_lines:
            logger.warning(
                f"Found OOM trace in dmesg: {oom_lines}, assuming OOM is the cause of realization kill."
            )
            return True
    return False


class Job:
    MEMORY_POLL_PERIOD = 5  # Seconds between memory polls

    def __init__(self, job_data, index, sleep_interval=1):
        self.sleep_interval = sleep_interval
        self.job_data = job_data
        self.index = index
        self.std_err = job_data.get("stderr")
        self.std_out = job_data.get("stdout")

    def run(self):
        start_message = Start(self)

        errors = self._check_job_files()

        errors.extend(self._assert_arg_list())

        self._dump_exec_env()

        if errors:
            yield start_message.with_error("\n".join(errors))
            return

        yield start_message

        executable = self.job_data.get("executable")
        assert_file_executable(executable)

        arg_list = [executable]
        if self.job_data.get("argList"):
            arg_list += self.job_data["argList"]

        # stdin/stdout/stderr are closed at the end of this function
        if self.job_data.get("stdin"):
            stdin = open(self.job_data.get("stdin"), encoding="utf-8")  # noqa
        else:
            stdin = None

        if self.std_err:
            os.makedirs(
                os.path.dirname(os.path.abspath(self.std_err)),
                exist_ok=True,
            )
            stderr = open(self.std_err, "w", encoding="utf-8")  # noqa
        else:
            stderr = None

        if self.std_out:
            os.makedirs(
                os.path.dirname(os.path.abspath(self.std_out)),
                exist_ok=True,
            )
            stdout = open(self.std_out, "w", encoding="utf-8")  # noqa
        else:
            stdout = None

        target_file = self.job_data.get("target_file")
        target_file_mtime: int = 0
        if target_file and os.path.exists(target_file):
            stat = os.stat(target_file)
            target_file_mtime = stat.st_mtime_ns

        max_running_minutes = self.job_data.get("max_running_minutes")
        run_start_time = dt.now()
        environment = self.job_data.get("environment")
        if environment is not None:
            environment = {**os.environ, **environment}

        def ensure_file_handles_closed():
            if stdin is not None:
                stdin.close()
            if stdout is not None:
                stdout.close()
            if stderr is not None:
                stderr.close()

        try:
            proc = Popen(
                arg_list,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                env=environment,
            )
            process = Process(proc.pid)
        except OSError as e:
            msg = f"{e.strerror} {e.filename}"
            if e.strerror == "Exec format error" and e.errno == 8:
                msg = (
                    f"Missing execution format information in file: {e.filename!r}."
                    f"Most likely you are missing and should add "
                    f"'#!/usr/bin/env python' to the top of the file: "
                )
            if stderr:
                stderr.write(msg)
            ensure_file_handles_closed()
            yield Exited(self, e.errno).with_error(msg)
            return

        exit_code = None

        # All child pids for the forward model step. Need to track these in order to be able
        # to detect OOM kills in case of failure.
        fm_step_pids = {process.pid}

        max_memory_usage = 0
        while exit_code is None:
            (memory_rss, cpu_seconds, oom_score) = _get_processtree_data(process)
            max_memory_usage = max(memory_rss, max_memory_usage)
            yield Running(
                self,
                ProcessTreeStatus(
                    rss=memory_rss,
                    max_rss=max_memory_usage,
                    fm_step_id=self.index,
                    fm_step_name=self.job_data.get("name"),
                    cpu_seconds=cpu_seconds,
                    oom_score=oom_score,
                ),
            )

            try:
                exit_code = process.wait(timeout=self.MEMORY_POLL_PERIOD)
            except TimeoutExpired:
                fm_step_pids |= {
                    int(child.pid) for child in process.children(recursive=True)
                }
                run_time = dt.now() - run_start_time
                if (
                    max_running_minutes is not None
                    and run_time.seconds > max_running_minutes * 60
                ):
                    # If the spawned process is not in the same process group as
                    # the callee (job_dispatch), we will kill the process group
                    # explicitly.
                    #
                    # Propagating the unsuccessful Exited message will kill the
                    # callee group. See job_dispatch.py.
                    process_group_id = os.getpgid(proc.pid)
                    this_group_id = os.getpgid(os.getpid())
                    if process_group_id != this_group_id:
                        os.killpg(process_group_id, signal.SIGKILL)

                    yield Exited(self, exit_code).with_error(
                        (
                            f"Job:{self.name()} has been running "
                            f"for more than {max_running_minutes} "
                            "minutes - explicitly killed."
                        )
                    )
                    return

        exited_message = Exited(self, exit_code)

        if exit_code != 0:
            if killed_by_oom(fm_step_pids):
                yield exited_message.with_error(
                    f"Forward model step {self.job_data.get('name')} "
                    "was killed due to out-of-memory. "
                    "Max memory usage recorded by Ert for the "
                    f"realization was {max_memory_usage//1024//1024} MB"
                )
            else:
                yield exited_message.with_error(
                    f"Process exited with status code {exit_code}"
                )
            return

        # exit_code is 0

        if self.job_data.get("error_file") and os.path.exists(
            self.job_data["error_file"]
        ):
            yield exited_message.with_error(
                f'Found the error file:{self.job_data["error_file"]} - job failed.'
            )
            return

        if target_file:
            target_file_error = self._check_target_file_is_written(target_file_mtime)
            if target_file_error:
                yield exited_message.with_error(target_file_error)
                return
        ensure_file_handles_closed()
        yield exited_message

    def _assert_arg_list(self):
        errors = []
        if "arg_types" in self.job_data:
            arg_types = self.job_data["arg_types"]
            arg_list = self.job_data.get("argList")
            for index, arg_type in enumerate(arg_types):
                if arg_type == "RUNTIME_FILE":
                    file_path = os.path.join(os.getcwd(), arg_list[index])
                    if not os.path.isfile(file_path):
                        errors.append(
                            f"In job {self.name()}: RUNTIME_FILE {arg_list[index]} "
                            "does not exist."
                        )
                if arg_type == "RUNTIME_INT":
                    try:
                        int(arg_list[index])
                    except ValueError:
                        errors.append(
                            (
                                f"In job {self.name()}: argument with index {index} "
                                "is of incorrect type, should be integer."
                            )
                        )
        return errors

    def name(self):
        return self.job_data["name"]

    def _dump_exec_env(self):
        exec_env = self.job_data.get("exec_env")
        if exec_env:
            exec_name, _ = os.path.splitext(
                os.path.basename(self.job_data.get("executable"))
            )
            with open(f"{exec_name}_exec_env.json", "w", encoding="utf-8") as f_handle:
                f_handle.write(json.dumps(exec_env, indent=4))

    def _check_job_files(self):
        """
        Returns the empty list if no failed checks, or a list of errors in case
        of failed checks.
        """
        errors = []
        if self.job_data.get("stdin") and not os.path.exists(self.job_data["stdin"]):
            errors.append(f'Could not locate stdin file: {self.job_data["stdin"]}')

        if self.job_data.get("start_file") and not os.path.exists(
            self.job_data["start_file"]
        ):
            errors.append(f'Could not locate start_file:{self.job_data["start_file"]}')

        if self.job_data.get("error_file") and os.path.exists(
            self.job_data.get("error_file")
        ):
            os.unlink(self.job_data.get("error_file"))

        return errors

    def _check_target_file_is_written(self, target_file_mtime: int, timeout=5):
        """
        Check whether or not a target_file eventually appear. Returns None in
        case of success, an error message in the case of failure.
        """
        # no target file is expected at all, indicate success
        if "target_file" not in self.job_data:
            return None

        target_file = self.job_data["target_file"]

        start_time = time.time()
        while True:
            if os.path.exists(target_file):
                stat = os.stat(target_file)
                if stat.st_mtime_ns > target_file_mtime:
                    return None

            time.sleep(self.sleep_interval)
            if time.time() - start_time > timeout:
                break

        # We have gone out of the loop via the break statement,
        # i.e. on a timeout.
        if os.path.exists(target_file):
            stat = os.stat(target_file)
            return (
                f"The target file:{target_file} has not been updated; "
                f"this is flagged as failure. mtime:{stat.st_mtime}   "
                f"stat_start_time:{target_file_mtime}"
            )
        return f"Could not find target_file:{target_file}"


def _get_processtree_data(
    process: Process,
) -> Tuple[int, float, Optional[int]]:
    """Obtain the oom_score (the Linux kernel uses this number to
    decide which process to kill first in out-of-memory siturations).

    Since the process being monitored here can have subprocesses using
    arbitrary memory amounts, we need to track the maximal oom_score for x
    all its descendants.

    oom_score defaults to 0 in Linux, but varies between -1000 and 1000.
    If returned value is None, then there is no information, e.g. if run
    on an OS not providing /proc/<pid>/oom_score

    Sum the memory measure RSS (resident set size) for a process and all
    its descendants.
    """

    oom_score = None
    # A value of None means that we have no information.
    memory_rss = 0
    cpu_seconds = 0.0
    with contextlib.suppress(ValueError, FileNotFoundError):
        oom_score = int(
            Path(f"/proc/{process.pid}/oom_score").read_text(encoding="utf-8")
        )
    with contextlib.suppress(
        ValueError, NoSuchProcess, AccessDenied, ZombieProcess, ProcessLookupError
    ), process.oneshot():
        memory_rss = process.memory_info().rss
        cpu_seconds = process.cpu_times().user

    with contextlib.suppress(
        NoSuchProcess, AccessDenied, ZombieProcess, ProcessLookupError
    ):
        for child in process.children(recursive=True):
            with contextlib.suppress(
                ValueError,
                FileNotFoundError,
                NoSuchProcess,
                AccessDenied,
                ZombieProcess,
                ProcessLookupError,
            ):
                oom_score_child = int(
                    Path(f"/proc/{child.pid}/oom_score").read_text(encoding="utf-8")
                )
                oom_score = (
                    max(oom_score, oom_score_child)
                    if oom_score is not None
                    else oom_score_child
                )
            with contextlib.suppress(
                NoSuchProcess, AccessDenied, ZombieProcess
            ), child.oneshot():
                memory_rss += child.memory_info().rss
                cpu_seconds += child.cpu_times().user
    return (memory_rss, cpu_seconds, oom_score)
