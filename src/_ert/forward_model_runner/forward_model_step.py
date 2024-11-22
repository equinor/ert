from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import signal
import socket
import sys
import time
from datetime import datetime as dt
from pathlib import Path
from subprocess import Popen, run
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Sequence, Tuple, cast

from psutil import AccessDenied, NoSuchProcess, Process, TimeoutExpired, ZombieProcess

from .io import check_executable
from .reporting.message import (
    Exited,
    ProcessTreeStatus,
    Running,
    Start,
)

if TYPE_CHECKING:
    from ert.config.forward_model_step import ForwardModelStepJSON

logger = logging.getLogger(__name__)


def killed_by_oom(pids: Sequence[int]) -> bool:
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


class ForwardModelStep:
    MEMORY_POLL_PERIOD = 5  # Seconds between memory polls

    def __init__(
        self, job_data: ForwardModelStepJSON, index: int, sleep_interval: int = 1
    ) -> None:
        self.sleep_interval = sleep_interval
        self.job_data = job_data
        self.index = index
        self.std_err = job_data.get("stderr")
        self.std_out = job_data.get("stdout")

    def run(self) -> Generator[Start | Exited | Running | None]:
        try:
            for msg in self._run():
                yield msg
        except Exception as e:
            yield Exited(self, exit_code=1).with_error(str(e))

    def create_start_message_and_check_job_files(self) -> Start:
        start_message = Start(self)

        errors = self._check_job_files()
        errors.extend(self._assert_arg_list())
        self._dump_exec_env()

        if errors:
            start_message = start_message.with_error("\n".join(errors))
        return start_message

    def _build_arg_list(self) -> List[str]:
        executable = self.job_data.get("executable")
        # assert executable is not None
        combined_arg_list = [executable]
        if arg_list := self.job_data.get("argList"):
            combined_arg_list += arg_list
        return combined_arg_list

    def _open_file_handles(
        self,
    ) -> Tuple[
        io.TextIOWrapper | None, io.TextIOWrapper | None, io.TextIOWrapper | None
    ]:
        if self.job_data.get("stdin"):
            stdin = open(cast(Path, self.job_data.get("stdin")), encoding="utf-8")  # noqa
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

    def _create_environment(self) -> Optional[Dict[str, str]]:
        combined_environment = None
        if environment := self.job_data.get("environment"):
            combined_environment = {**os.environ, **environment}
        return combined_environment

    def _run(self) -> Generator[Start | Exited | Running | None]:
        start_message = self.create_start_message_and_check_job_files()

        yield start_message
        if not start_message.success():
            return

        arg_list = self._build_arg_list()

        (stdin, stdout, stderr) = self._open_file_handles()
        # stdin/stdout/stderr are closed at the end of this function

        target_file = self.job_data.get("target_file")
        target_file_mtime: Optional[int] = _get_target_file_ntime(target_file)

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
                potential_exited_msg = (
                    self.handle_process_timeout_and_create_exited_msg(exit_code, proc)
                )
                if isinstance(potential_exited_msg, Exited):
                    yield potential_exited_msg

                    return
                fm_step_pids |= {
                    int(child.pid) for child in process.children(recursive=True)
                }

        ensure_file_handles_closed([stdin, stdout, stderr])
        exited_message = self._create_exited_message_based_on_exit_code(
            max_memory_usage, target_file_mtime, exit_code, fm_step_pids
        )
        yield exited_message

    def _create_exited_message_based_on_exit_code(
        self,
        max_memory_usage: int,
        target_file_mtime: Optional[int],
        exit_code: int,
        fm_step_pids: Sequence[int],
    ) -> Exited:
        if exit_code != 0:
            exited_message = self._create_exited_msg_for_non_zero_exit_code(
                max_memory_usage, exit_code, fm_step_pids
            )
            return exited_message

        exited_message = Exited(self, exit_code)
        if self.job_data.get("error_file") and os.path.exists(
            self.job_data["error_file"]
        ):
            return exited_message.with_error(
                f'Found the error file:{self.job_data["error_file"]} - job failed.'
            )

        if target_file_mtime:
            target_file_error = self._check_target_file_is_written(target_file_mtime)
            if target_file_error:
                return exited_message.with_error(target_file_error)

        return exited_message

    def _create_exited_msg_for_non_zero_exit_code(
        self,
        max_memory_usage: int,
        exit_code: int,
        fm_step_pids: Sequence[int],
    ) -> Exited:
        # All child pids for the forward model step. Need to track these in order to be able
        # to detect OOM kills in case of failure.
        exited_message = Exited(self, exit_code)

        if killed_by_oom(fm_step_pids):
            return exited_message.with_error(
                f"Forward model step {self.job_data.get('name')} "
                f"was killed due to out-of-memory on {socket.gethostname()}. "
                "Max memory usage recorded by Ert for the "
                f"realization was {max_memory_usage//1024//1024} MB. "
                "Please add REALIZATION_MEMORY to your ert config together "
                "with a suitable memory amount to avoid this."
            )
        return exited_message.with_error(
            f"Process exited with status code {exited_message.exit_code}"
        )

    def handle_process_timeout_and_create_exited_msg(
        self, exit_code: Optional[int], proc: Popen[Process]
    ) -> Exited | None:
        max_running_minutes = self.job_data.get("max_running_minutes")
        run_start_time = dt.now()

        run_time = dt.now() - run_start_time
        if max_running_minutes is None or run_time.seconds > max_running_minutes * 60:
            return None

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

        return Exited(self, exit_code).with_error(
            (
                f"Job:{self.name()} has been running "
                f"for more than {max_running_minutes} "
                "minutes - explicitly killed."
            )
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
        return self.job_data["name"]

    def _dump_exec_env(self) -> None:
        exec_env = self.job_data.get("exec_env")
        if exec_env:
            exec_name, _ = os.path.splitext(
                os.path.basename(cast(Path, self.job_data.get("executable")))
            )
            with open(f"{exec_name}_exec_env.json", "w", encoding="utf-8") as f_handle:
                f_handle.write(json.dumps(exec_env, indent=4))

    def _check_job_files(self) -> list[str]:
        """
        Returns the empty list if no failed checks, or a list of errors in case
        of failed checks.
        """
        errors = []
        if self.job_data.get("stdin") and not os.path.exists(self.job_data["stdin"]):
            errors.append(f'Could not locate stdin file: {self.job_data["stdin"]}')

        if self.job_data.get("start_file") and not os.path.exists(
            cast(Path, self.job_data["start_file"])
        ):
            errors.append(f'Could not locate start_file:{self.job_data["start_file"]}')

        if self.job_data.get("error_file") and os.path.exists(
            cast(Path, self.job_data.get("error_file"))
        ):
            os.unlink(cast(Path, self.job_data.get("error_file")))

        if executable_error := check_executable(self.job_data.get("executable")):
            errors.append(executable_error)

        return errors

    def _check_target_file_is_written(
        self, target_file_mtime: int, timeout: int = 5
    ) -> str | None:
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


def _get_target_file_ntime(file: Optional[str]) -> Optional[int]:
    mtime = None
    if file and os.path.exists(file):
        stat = os.stat(file)
        mtime = stat.st_mtime_ns
    return mtime


def ensure_file_handles_closed(file_handles: Sequence[io.TextIOWrapper | None]) -> None:
    for file_handle in file_handles:
        if file_handle is not None:
            file_handle.close()


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
    with (
        contextlib.suppress(
            ValueError, NoSuchProcess, AccessDenied, ZombieProcess, ProcessLookupError
        ),
        process.oneshot(),
    ):
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
            with (
                contextlib.suppress(NoSuchProcess, AccessDenied, ZombieProcess),
                child.oneshot(),
            ):
                memory_rss += child.memory_info().rss
                cpu_seconds += child.cpu_times().user
    return (memory_rss, cpu_seconds, oom_score)
