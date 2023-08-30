import json
import os
import signal
import time
from datetime import datetime as dt
from subprocess import Popen
from typing import Dict, Iterator, List, Optional, TypedDict

from psutil import AccessDenied, NoSuchProcess, Process, TimeoutExpired, ZombieProcess

from _ert_job_runner.io import assert_file_executable
from _ert_job_runner.reporting.message import Exited, Message, Running, Start


class JobData(TypedDict):
    stderr: str
    stdout: str
    stdin: str
    name: str
    start_file: str
    error_file: str
    target_file: str
    executable: str
    argList: List[str]
    arg_types: List[str]
    max_running_minutes: int
    environment: Dict[str, str]


class Job:
    MEMORY_POLL_PERIOD = 5  # Seconds between memory polls

    def __init__(self, job_data: JobData, index: int, sleep_interval: int = 1) -> None:
        self.sleep_interval = sleep_interval
        self.job_data = job_data
        self.index = index
        self.std_err = job_data.get("stderr")
        self.std_out = job_data.get("stdout")

    def run(self) -> Iterator[Message]:
        # pylint: disable=consider-using-with
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
        assert executable is not None

        arg_list = [executable]
        if self.job_data.get("argList"):
            arg_list += self.job_data["argList"]

        # pylint: disable=consider-using-with
        # stdin/stdout/stderr are closed at the end of this function
        if self.job_data.get("stdin"):
            stdin = open(self.job_data["stdin"], encoding="utf-8")
        else:
            stdin = None

        if self.std_err:
            os.makedirs(
                os.path.dirname(os.path.abspath(self.std_err)),
                exist_ok=True,
            )
            stderr = open(self.std_err, "w", encoding="utf-8")
        else:
            stderr = None

        if self.std_out:
            os.makedirs(
                os.path.dirname(os.path.abspath(self.std_out)),
                exist_ok=True,
            )
            stdout = open(self.std_out, "w", encoding="utf-8")
        else:
            stdout = None

        target_file = self.job_data.get("target_file")
        if target_file:
            target_file_mtime: int = 0
            if os.path.exists(target_file):
                stat = os.stat(target_file)
                target_file_mtime = stat.st_mtime_ns

        max_running_minutes = self.job_data.get("max_running_minutes")
        run_start_time = dt.now()
        environment = self.job_data.get("environment")
        if environment is not None:
            environment = {**os.environ, **environment}

        def ensure_file_handles_closed() -> None:
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

        max_memory_usage = 0
        while exit_code is None:
            try:
                memory = process.memory_info().rss
            except (NoSuchProcess, AccessDenied, ZombieProcess):
                # In case of a process that has died and is in some transitional
                # state, we ignore any failures. Only seen on OSX thus far.
                #
                # See https://github.com/giampaolo/psutil/issues/1044#issuecomment-298745532  # noqa
                memory = 0
            if memory > max_memory_usage:
                max_memory_usage = memory

            yield Running(self, max_memory_usage, memory)

            try:
                exit_code = process.wait(timeout=self.MEMORY_POLL_PERIOD)
            except TimeoutExpired:
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

    def _assert_arg_list(self) -> List[str]:
        errors = []
        if "arg_types" in self.job_data and "argList" in self.job_data:
            arg_types = self.job_data["arg_types"]
            arg_list = self.job_data["argList"]
            for arg, arg_type in zip(arg_list, arg_types):
                if arg_type == "RUNTIME_FILE":
                    file_path = os.path.join(os.getcwd(), arg)
                    if not os.path.isfile(file_path):
                        errors.append(
                            f"In job {self.name()}: RUNTIME_FILE {arg} "
                            "does not exist."
                        )
                if arg_type == "RUNTIME_INT":
                    try:
                        int(arg)
                    except ValueError:
                        errors.append(
                            (
                                f"In job {self.name()}: argument {arg} "
                                "is of incorrect type, should be integer."
                            )
                        )
        return errors

    def name(self) -> str:
        return self.job_data["name"]

    def _dump_exec_env(self) -> None:
        exec_env = self.job_data.get("exec_env")
        if exec_env:
            exec_name, _ = os.path.splitext(
                os.path.basename(self.job_data["executable"])
            )
            with open(f"{exec_name}_exec_env.json", "w", encoding="utf-8") as f_handle:
                f_handle.write(json.dumps(exec_env, indent=4))

    def _check_job_files(self) -> List[str]:
        """
        Returns the empty list if no failed checks, or a list of errors in case
        of failed checks.
        """
        errors = []
        if "stdin" in self.job_data and not os.path.exists(self.job_data["stdin"]):
            errors.append(f'Could not locate stdin file: {self.job_data["stdin"]}')

        if "start_file" in self.job_data and not os.path.exists(
            self.job_data["start_file"]
        ):
            errors.append(f'Could not locate start_file:{self.job_data["start_file"]}')

        if "error_file" in self.job_data and os.path.exists(
            self.job_data["error_file"]
        ):
            os.unlink(self.job_data["error_file"])

        return errors

    def _check_target_file_is_written(
        self, target_file_mtime: int, timeout: int = 5
    ) -> Optional[str]:
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
