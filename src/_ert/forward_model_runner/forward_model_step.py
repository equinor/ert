from __future__ import annotations

import contextlib
import io
import logging
import os
import signal
import socket
import sys
import time
import uuid
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from subprocess import Popen, run
from typing import TYPE_CHECKING, cast

from psutil import AccessDenied, NoSuchProcess, Process, TimeoutExpired, ZombieProcess

from .io import check_executable
from .reporting.message import (
    Exited,
    ProcessTreeStatus,
    Running,
    Start,
)

if TYPE_CHECKING:
    from ert.config import ForwardModelStepJSON

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
        dmesg_result = run(
            ["dmesg", "-T", "--time-format=iso"], capture_output=True, check=False
        )
        if dmesg_result.returncode != 0:
            logger.warning(
                "Could not use dmesg to check for OOM kill, "
                f"returncode {dmesg_result.returncode} "
                f"and stderr: {dmesg_result.stderr}"  # type: ignore
            )
            return False
    except FileNotFoundError:
        logger.warning(
            "Could not use dmesg to check for OOM kill, utility not available"
        )
        return False

    oom_lines = "\n".join(
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
                f"Found OOM trace in dmesg: \n{oom_lines}\n"
                "assuming OOM is the cause of realization kill."
            )
            return True
    return False


class ForwardModelStep:
    MEMORY_POLL_PERIOD = 5  # Seconds between memory polls
    TARGET_FILE_POLL_PERIOD = 5  # Seconds to wait for target file after step completion

    def __init__(
        self, step_data: ForwardModelStepJSON, index: int, sleep_interval: int = 1
    ) -> None:
        self.sleep_interval = sleep_interval
        self.step_data = step_data
        self.index = index
        self.std_err = step_data.get("stderr")
        self.std_out = step_data.get("stdout")

    def run(self) -> Generator[Start | Exited | Running]:
        try:
            yield from self._run()
        except Exception as e:
            yield Exited(self, exit_code=1).with_error(str(e))

    def create_start_message_and_check_step_files(self) -> Start:
        start_message = Start(self)

        errors = self._check_step_files()
        errors.extend(self._assert_arg_list())

        if errors:
            start_message = start_message.with_error("\n".join(errors))
        return start_message

    def _build_arg_list(self) -> list[str]:
        executable = self.step_data.get("executable")
        combined_arg_list = [executable]
        if arg_list := self.step_data.get("argList"):
            combined_arg_list += arg_list
        return combined_arg_list  # type: ignore

    def _open_file_handles(
        self,
    ) -> tuple[
        io.TextIOWrapper | None, io.TextIOWrapper | None, io.TextIOWrapper | None
    ]:
        if self.step_data.get("stdin"):
            stdin = open(cast(Path, self.step_data.get("stdin")), encoding="utf-8")  # noqa
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

        return (stdin, stdout, stderr)

    def _create_environment(self) -> dict[str, str] | None:
        combined_environment = None
        if environment := self.step_data.get("environment"):
            combined_environment = {**os.environ, **environment}
        return combined_environment

    def _run(self) -> Generator[Start | Exited | Running]:
        start_message = self.create_start_message_and_check_step_files()

        yield start_message
        if not start_message.success():
            return

        arg_list = self._build_arg_list()

        (stdin, stdout, stderr) = self._open_file_handles()
        # stdin/stdout/stderr are closed at the end of this function

        target_file: str | None = self.step_data.get("target_file")
        existing_target_file_mtime: int | None = _get_existing_target_file_mtime(
            target_file
        )
        run_start_time = dt.now()
        try:
            proc = Popen(
                arg_list,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                env=self._create_environment(),
            )
            process = Process(proc.pid)
        except OSError as e:
            exited_message = self._handle_process_io_error_and_create_exited_message(
                e, stderr
            )
            yield exited_message
            ensure_file_handles_closed([stdin, stdout, stderr])
            return

        exit_code = None

        max_memory_usage = 0
        fm_step_pids = {int(process.pid)}
        cpu_seconds_processtree: ProcesstreeTimer = ProcesstreeTimer()
        while True:
            try:
                exit_code = process.wait(timeout=self.MEMORY_POLL_PERIOD)
                if exit_code is not None:
                    break
            except TimeoutExpired:
                potential_exited_msg = (
                    self.handle_process_timeout_and_create_exited_msg(
                        exit_code,
                        proc,  # type: ignore
                        run_start_time,
                    )
                )
                if isinstance(potential_exited_msg, Exited):
                    yield potential_exited_msg
                    return

            (memory_rss, cpu_seconds_snapshot, oom_score, pids) = _get_processtree_data(
                process
            )
            cpu_seconds_processtree.update(cpu_seconds_snapshot)
            fm_step_pids |= pids
            max_memory_usage = max(memory_rss, max_memory_usage)
            yield Running(
                self,
                ProcessTreeStatus(
                    rss=memory_rss,
                    max_rss=max_memory_usage,
                    fm_step_id=self.index,
                    fm_step_name=self.step_data.get("name"),
                    cpu_seconds=cpu_seconds_processtree.total_cpu_seconds(),
                    oom_score=oom_score,
                ),
            )
        ensure_file_handles_closed([stdin, stdout, stderr])
        exited_message = self._create_exited_message_based_on_exit_code(
            max_memory_usage,
            target_file,
            existing_target_file_mtime,
            exit_code,
            fm_step_pids,
        )
        yield exited_message

    def _create_exited_message_based_on_exit_code(
        self,
        max_memory_usage: int,
        target_file: str | None,
        existing_target_file_mtime: int | None,
        exit_code: int,
        fm_step_pids: set[int],
    ) -> Exited:
        if exit_code != 0:
            exited_message = self._create_exited_msg_for_non_zero_exit_code(
                max_memory_usage, exit_code, fm_step_pids
            )
            return exited_message

        exited_message = Exited(self, exit_code)
        if self.step_data.get("error_file") and os.path.exists(
            self.step_data["error_file"]
        ):
            return exited_message.with_error(
                f"Found the error file:{self.step_data['error_file']} - step failed."
            )

        if target_file:
            target_file_error = self._check_target_file_is_written(
                target_file, existing_target_file_mtime, self.TARGET_FILE_POLL_PERIOD
            )
            if target_file_error:
                return exited_message.with_error(target_file_error)

        return exited_message

    def _create_exited_msg_for_non_zero_exit_code(
        self,
        max_memory_usage: int,
        exit_code: int,
        fm_step_pids: set[int],
    ) -> Exited:
        # All child pids for the forward model step. Need to track these in order
        # to be able to detect OOM kills in case of failure.
        exited_message = Exited(self, exit_code)

        if killed_by_oom(fm_step_pids):
            return exited_message.with_error(
                f"Forward model step {self.step_data.get('name')} "
                f"was killed due to out-of-memory on {socket.gethostname()}. "
                "Max memory usage recorded by Ert for the "
                f"realization was {max_memory_usage // 1024 // 1024} MB. "
                "Please add REALIZATION_MEMORY to your ert config together "
                "with a suitable memory amount to avoid this."
            )
        return exited_message.with_error(
            f"Process exited with status code {exited_message.exit_code}"
        )

    def handle_process_timeout_and_create_exited_msg(
        self,
        exit_code: int | None,
        proc: Popen[Process],  # type: ignore
        run_start_time: dt,
    ) -> Exited | None:
        max_running_minutes = self.step_data.get("max_running_minutes")

        run_time = dt.now() - run_start_time
        if max_running_minutes is None or run_time.seconds < max_running_minutes * 60:
            return None

        # If the spawned process is not in the same process group as
        # the callee (fm_dispatch), we will kill the process group
        # explicitly.
        #
        # Propagating the unsuccessful Exited message will kill the
        # callee group. See fm_dispatch.py.
        process_group_id = os.getpgid(proc.pid)
        this_group_id = os.getpgid(os.getpid())
        if process_group_id != this_group_id:
            os.killpg(process_group_id, signal.SIGKILL)

        return Exited(self, exit_code).with_error(
            f"Step:{self.name()} has been running "
            f"for more than {max_running_minutes} "
            "minutes - explicitly killed."
        )

    def _handle_process_io_error_and_create_exited_message(
        self, e: OSError, stderr: io.TextIOWrapper | None
    ) -> Exited:
        msg = f"{e.strerror} {e.filename}"
        if e.strerror == "Exec format error" and e.errno == 8:
            msg = (
                f"Missing execution format information in file: {e.filename!r}."
                f"Most likely you are missing and should add "
                f"'#!/usr/bin/env python' to the top of the file: "
            )
        if stderr:
            stderr.write(msg)
        return Exited(self, e.errno).with_error(msg)

    def name(self) -> str:
        return self.step_data["name"]

    def _check_step_files(self) -> list[str]:
        """
        Returns the empty list if no failed checks, or a list of errors in case
        of failed checks.
        """
        errors = []
        if self.step_data.get("stdin") and not os.path.exists(self.step_data["stdin"]):
            errors.append(f"Could not locate stdin file: {self.step_data['stdin']}")

        if self.step_data.get("start_file") and not os.path.exists(
            cast(Path, self.step_data["start_file"])
        ):
            errors.append(f"Could not locate start_file:{self.step_data['start_file']}")

        if self.step_data.get("error_file") and os.path.exists(
            cast(Path, self.step_data.get("error_file"))
        ):
            os.unlink(cast(Path, self.step_data.get("error_file")))

        if executable_error := check_executable(self.step_data.get("executable")):
            errors.append(executable_error)

        return errors

    def _check_target_file_is_written(
        self, target_file: str, existing_target_file_mtime: int | None, timeout: int = 5
    ) -> str | None:
        """
        Check whether or not a target_file eventually appear. Returns None in
        case of success, an error message in the case of failure.
        """

        start_time = time.time()
        while True:
            if os.path.exists(target_file):
                stat = os.stat(target_file)
                if stat.st_mtime_ns > (existing_target_file_mtime or 0):
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
                f"stat_start_time:{existing_target_file_mtime}"
            )
        return f"Could not find target_file:{target_file}"

    def _assert_arg_list(self) -> list[str]:
        errors: list[str] = []
        if "arg_types" in self.step_data:  # This seems to be NEVER true(?)
            arg_types = self.step_data["arg_types"]  # type: ignore
            arg_list = self.step_data.get("argList", [])
            for index, arg_type in enumerate(arg_types):
                if arg_type == "RUNTIME_FILE":
                    file_path = os.path.join(os.getcwd(), arg_list[index])
                    if not os.path.isfile(file_path):
                        errors.append(
                            f"In step {self.name()}: RUNTIME_FILE {arg_list[index]} "
                            "does not exist."
                        )
                if arg_type == "RUNTIME_INT":
                    try:
                        int(arg_list[index])
                    except ValueError:
                        errors.append(
                            f"In step {self.name()}: argument with index {index} "
                            "is of incorrect type, should be integer."
                        )
        return errors


def _get_existing_target_file_mtime(file: str | None) -> int | None:
    mtime = None
    if file and os.path.exists(file):
        stat = os.stat(file)
        mtime = stat.st_mtime_ns
    return mtime


def ensure_file_handles_closed(file_handles: Sequence[io.TextIOWrapper | None]) -> None:
    for file_handle in file_handles:
        if file_handle is not None:
            file_handle.close()


@dataclass
class ProcesstreeTimer:
    _cpu_seconds_pr_pid: dict[str, float] = field(default_factory=dict, init=False)

    def update(self, cpu_seconds_snapshot: dict[str, float]) -> None:
        for pid, seconds in cpu_seconds_snapshot.items():
            if self._cpu_seconds_pr_pid.get(pid, 0.0) > seconds:
                # cpu_seconds for a process must increase monotonically.
                # Since decreasing cpu_seconds was detected, it must be due to pid reuse
                self._cpu_seconds_pr_pid[pid + "-" + str(uuid.uuid4())] = (
                    self._cpu_seconds_pr_pid[pid]
                )
            self._cpu_seconds_pr_pid[pid] = seconds

    def total_cpu_seconds(self) -> float:
        return sum(self._cpu_seconds_pr_pid.values())


def _get_processtree_data(
    process: Process,
) -> tuple[int, dict[str, float], int | None, set[int]]:
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
    cpu_seconds_pr_pid: dict[str, float] = {}
    pids = set()
    with contextlib.suppress(ValueError, FileNotFoundError):
        oom_score = int(
            Path(f"/proc/{process.pid}/oom_score").read_text(encoding="utf-8")
        )
    with (
        contextlib.suppress(
            ValueError, NoSuchProcess, AccessDenied, ZombieProcess, ProcessLookupError
        ),
        process.oneshot(),
    ):
        memory_rss = process.memory_info().rss
        cpu_seconds_pr_pid[str(process.pid)] = process.cpu_times().user

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
                pids.add(int(child.pid))
            with (
                contextlib.suppress(NoSuchProcess, AccessDenied, ZombieProcess),
                child.oneshot(),
            ):
                memory_rss += child.memory_info().rss
                cpu_seconds_pr_pid[str(child.pid)] = child.cpu_times().user
    return (memory_rss, cpu_seconds_pr_pid, oom_score, pids)
