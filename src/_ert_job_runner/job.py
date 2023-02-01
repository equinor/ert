import json
import os
import signal
import time
from datetime import datetime as dt
from subprocess import Popen

from psutil import AccessDenied, NoSuchProcess, Process, TimeoutExpired, ZombieProcess

from _ert_job_runner.io import assert_file_executable
from _ert_job_runner.reporting.message import Exited, Running, Start


class Job:
    MEMORY_POLL_PERIOD = 5  # Seconds between memory polls

    def __init__(self, job_data, index, sleep_interval=1):
        self.sleep_interval = sleep_interval
        self.job_data = job_data
        self.index = index
        self.std_err = job_data.get("stderr")
        self.std_out = job_data.get("stdout")

    def run(self):
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

        arg_list = [executable]
        if self.job_data.get("argList"):
            arg_list += self.job_data["argList"]

        # pylint: disable=consider-using-with
        # stdin/stdout/stderr are closed at the end of this function
        if self.job_data.get("stdin"):
            stdin = open(self.job_data.get("stdin"), encoding="utf-8")
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

        exec_env = self.job_data.get("exec_env")
        if exec_env:
            exec_name, _ = os.path.splitext(
                os.path.basename(self.job_data.get("executable"))
            )
            with open(f"{exec_name}_exec_env.json", "w", encoding="utf-8") as f_handle:
                f_handle.write(json.dumps(exec_env, indent=4))

        max_running_minutes = self.job_data.get("max_running_minutes")
        run_start_time = dt.now()

        proc = Popen(
            arg_list,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=self.job_data.get("environment"),
        )

        exit_code = None

        process = Process(proc.pid)
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

        if stdin is not None:
            stdin.close()
        if stdout is not None:
            stdout.close()
        if stderr is not None:
            stderr.close()

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
                f_handle.write(json.dumps(exec_env))

    def _check_job_files(self):
        """
        Returns the empty list if no failed checks, or a list of errors in case
        of failed checks.
        """
        errors = []
        if self.job_data.get("stdin"):
            if not os.path.exists(self.job_data["stdin"]):
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
