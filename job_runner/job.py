import json
import os
import signal
import time
from datetime import datetime as dt
from subprocess import Popen

from psutil import Process, TimeoutExpired, NoSuchProcess, AccessDenied, ZombieProcess

from job_runner.io import assert_file_executable
from job_runner.reporting.message import Exited, Running, Start


class Job(object):
    MEMORY_POLL_PERIOD = 5  # Seconds between memory polls

    def __init__(self, job_data, index, sleep_interval=1):
        self.sleep_interval = sleep_interval
        self.job_data = job_data
        self.index = index
        self.std_err = None
        self.std_out = None
        if "stderr" in job_data and job_data["stderr"]:
            self.std_err = job_data["stderr"]
        if "stdout" in job_data and job_data["stdout"]:
            self.std_out = job_data["stdout"]

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

        if self.job_data.get("stdin"):
            stdin = open(self.job_data.get("stdin"))
        else:
            stdin = None

        if self.std_err:
            stderr = open(self.std_err, "w")
        else:
            stderr = None

        if self.std_out:
            stdout = open(self.std_out, "w")
        else:
            stdout = None

        if self.job_data.get("target_file"):
            target_file_mtime = 0
            if os.path.exists(self.job_data["target_file"]):
                stat = os.stat(self.job_data["target_file"])
                target_file_mtime = stat.st_mtime

        exec_env = self.job_data.get("exec_env")
        if exec_env:
            exec_name, _ = os.path.splitext(
                os.path.basename(self.job_data.get("executable"))
            )
            with open("{}_exec_env.json".format(exec_name), "w") as f:
                f.write(json.dumps(exec_env))

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
                """In case of a process that has died and is in some
                transitional state, we ignore any failures. Only seen on OSX
                thus far.
                See https://github.com/giampaolo/psutil/issues/1044#issuecomment-298745532
                """
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
                    """
                    If the spawned process is not in the same process group
                    as the callee (job_dispatch), we will kill the process
                    group explicitly.

                    Propagating the unsuccessful Exited message will kill the
                    callee group. See job_dispatch.py.
                    """
                    process_group_id = os.getpgid(proc.pid)
                    this_group_id = os.getpgid(os.getpid())
                    if process_group_id != this_group_id:
                        os.killpg(process_group_id, signal.SIGKILL)

                    yield Exited(self, exit_code).with_error(
                        "Job:{} has been running for more than {} minutes - explicitly killed.".format(
                            self.name(), max_running_minutes
                        )
                    )
                    return

        exited_message = Exited(self, exit_code)

        if exit_code != 0:
            yield exited_message.with_error(
                "Process exited with status code {}".format(exit_code)
            )
            return

        # exit_code is 0

        if self.job_data.get("error_file"):
            if os.path.exists(self.job_data["error_file"]):
                yield exited_message.with_error(
                    "Found the error file:{} - job failed.".format(
                        self.job_data["error_file"]
                    )
                )
                return

        if self.job_data.get("target_file"):
            target_file_error = self._check_target_file_is_written(target_file_mtime)
            if target_file_error:
                yield exited_message.with_error(target_file_error)
                return

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
                            "In job {}: RUNTIME_FILE {} does not exist.".format(
                                self.name(), arg_list[index]
                            )
                        )
                if arg_type == "RUNTIME_INT":
                    try:
                        int(arg_list[index])
                    except ValueError:
                        errors.append(
                            "In job {}: argument with index {} is of incorrect type, should be integer.".format(
                                self.name(), index
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
            with open("{}_exec_env.json".format(exec_name), "w") as f:
                f.write(json.dumps(exec_env))

    def _check_job_files(self):
        """
        Returns the empty list if no failed checks, or a list of errors in case
        of failed checks.
        """
        errors = []
        if self.job_data.get("stdin"):
            if not os.path.exists(self.job_data["stdin"]):
                errors.append(
                    "Could not locate stdin file: {}".format(self.job_data["stdin"])
                )

        if self.job_data.get("start_file"):
            if not os.path.exists(self.job_data["start_file"]):
                errors.append(
                    "Could not locate start_file:{}".format(self.job_data["start_file"])
                )

        if self.job_data.get("error_file"):
            if os.path.exists(self.job_data.get("error_file")):
                os.unlink(self.job_data.get("error_file"))

        return errors

    def _check_target_file_is_written(self, target_file_mtime, timeout=5):
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
                if stat.st_mtime > target_file_mtime:
                    return None

            time.sleep(self.sleep_interval)
            if time.time() - start_time > timeout:
                break

        # We have gone out of the loop via the break statement,
        # i.e. on a timeout.
        if os.path.exists(target_file):
            stat = os.stat(target_file)
            return "The target file:{} has not been updated; this is flagged as failure. mtime:{}   stat_start_time:{}".format(
                target_file, stat.st_mtime, target_file_mtime
            )
        else:
            return "Could not find target_file:{}".format(target_file)
