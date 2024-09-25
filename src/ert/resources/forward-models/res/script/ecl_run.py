import datetime
import glob
import os
import os.path
import re
import socket
import subprocess
import sys
import time
from argparse import ArgumentParser
from collections import namedtuple
from contextlib import contextmanager, suppress
from pathlib import Path
from random import random
from typing import List

import resfo
from ecl_config import EclConfig, EclrunConfig, Simulator
from packaging import version


def ecl_output_has_license_error(ecl_output: str):
    return (
        "LICENSE ERROR" in ecl_output
        or "LICENSE FAILURE" in ecl_output
        or "not allowed in license" in ecl_output
    )


class EclError(RuntimeError):
    def failed_due_to_license_problems(self) -> bool:
        # self.args[0] contains the multiline ERROR messages and SLAVE startup messages
        if ecl_output_has_license_error(self.args[0]):
            return True
        if re.search(a_slave_failed_pattern, self.args[0]):
            for match in re.finditer(slave_run_paths, self.args[0], re.MULTILINE):
                (ecl_case_starts_with, ecl_case_dir) = match.groups()
                for prt_file in glob.glob(
                    f"{ecl_case_dir}/{ecl_case_starts_with}*.PRT"
                ):
                    if ecl_output_has_license_error(
                        Path(prt_file).read_text(encoding="utf-8")
                    ):
                        return True
        return False


def await_process_tee(process, *out_files) -> int:
    """Wait for process to finish, "tee"-ing the subprocess' stdout into all the
    given file objects.

    NB: We aren't checking if `os.write` succeeds. It succeeds if its return
    value matches `len(bytes_)`. In other cases we might want to do something
    smart, such as retry or raise an error. At the time of writing it is
    uncertain what we should do, and it is assumed that data loss is acceptable.

    """
    out_fds = [f.fileno() for f in out_files]
    process_fd = process.stdout.fileno()

    while True:
        while True:
            bytes_ = os.read(process_fd, 4096)
            if bytes_ == b"":  # check EOF
                break
            for fd in out_fds:
                os.write(fd, bytes_)

        # Check if process terminated
        if process.poll() is not None:
            break
    process.stdout.close()

    return process.returncode


EclipseResult = namedtuple("EclipseResult", "errors bugs")
body_sub_pattern = r"(\s^\s@.+$)*"
date_sub_pattern = r"\s+AT TIME\s+(?P<Days>\d+\.\d+)\s+DAYS\s+\((?P<Date>(.+)):\s*$"
error_pattern_e100 = rf"^\s@--  ERROR{date_sub_pattern}${body_sub_pattern}"
error_pattern_e300 = rf"^\s@--Error${body_sub_pattern}"
slave_started_pattern = (
    rf"^\s@--MESSAGE{date_sub_pattern}\s^\s@\s+STARTING SLAVE.+${body_sub_pattern}"
)
a_slave_failed_pattern = r"\s@\s+SLAVE RUN.*HAS STOPPED WITH AN ERROR CONDITION.\s*"
slave_run_paths = r"^\s@\s+STARTING SLAVE\s+[^ ]+RUNNING \([^ ]\)\s*$"
slave_run_paths = r"\s@\s+STARTING SLAVE .* RUNNING (\w+)\s*^\s@\s+ON HOST.*IN DIRECTORY\s*^\s@\s+(.*)"


def make_LSB_MCPU_machine_list(LSB_MCPU_HOSTS):
    host_numcpu_list = LSB_MCPU_HOSTS.split()
    host_list = []
    for index in range(len(host_numcpu_list) // 2):
        machine = host_numcpu_list[2 * index]
        host_numcpu = int(host_numcpu_list[2 * index + 1])
        for _ in range(host_numcpu):
            host_list.append(machine)
    return host_list


def _expand_SLURM_range(rs):
    if "-" in rs:
        tmp = rs.split("-")
        return range(int(tmp[0]), int(tmp[1]) + 1)
    else:
        return [int(rs)]


def _expand_SLURM_node(node_string):
    match_object = re.match(r"(?P<base>[^[]+)\[(?P<range>[-0-9,]+)\]", node_string)
    if match_object:
        node_list = []
        base = match_object.groupdict()["base"]
        range_string = match_object.groupdict()["range"]
        for rs in range_string.split(","):
            for num in _expand_SLURM_range(rs):
                node_list.append(f"{base}{num}")
        return node_list
    else:
        return [node_string]


def _expand_SLURM_task_count(task_count_string):
    match_object = re.match(r"(?P<count>\d+)(\(x(?P<mult>\d+)\))?", task_count_string)
    if match_object:
        match_dict = match_object.groupdict()
        print(match_dict)
        count = int(match_dict["count"])
        mult_string = match_dict["mult"]
        mult = 1 if mult_string is None else int(mult_string)

        return [count] * mult
    else:
        raise ValueError(f"Failed to parse SLURM_TASKS_PER_NODE: {task_count_string}")


# The list of available machines/nodes and how many tasks each node should get
# is available in the slurm environment variables SLURM_JOB_NODELIST and
# SLURM_TASKS_PER_NODE. These string variables are in an incredibly compact
# notation, and there are some hoops to expand them. The short description is:
#
#  1. They represent flat lists of hostnames and the number of cpu's on that
#     host respectively.
#
#  2. The outer structure is a ',' separated lis.
#
#  3. The items in SLURM_JOB_NODELIST have a compact notation
#     base-[n1-n2,n3-n4] which is expanded to the nodelist: [base-n1,
#     base-n1+1, base-n1+2, ... , base-n4-1, base-n4]
#
#  4. The SLURM_TASK_PER_NODE items has the compact notation 3(x4) which
#     implies that four consecutive nodes (from the expanded
#     SLURM_JOB_NODELIST) should have three CPUs each.
#
# For further details see the sbatch manual page.


def make_SLURM_machine_list(SLURM_JOB_NODELIST, SLURM_TASKS_PER_NODE):
    # We split on ',' - but not on ',' which is inside a [...]
    split_re = ",(?![^[]*\\])"
    nodelist = []
    for node_string in re.split(split_re, SLURM_JOB_NODELIST):
        nodelist += _expand_SLURM_node(node_string)

    task_count_list = []
    for task_count_string in SLURM_TASKS_PER_NODE.split(","):
        task_count_list += _expand_SLURM_task_count(task_count_string)

    host_list = []
    for node, count in zip(nodelist, task_count_list):
        host_list += [node] * count

    return host_list


def make_LSB_machine_list(LSB_HOSTS):
    return LSB_HOSTS.split()


@contextmanager
def pushd(run_path):
    starting_directory = os.getcwd()
    os.chdir(run_path)
    yield
    os.chdir(starting_directory)


def _find_unsmry(case: str) -> str:
    def _is_unsmry(base: str, path: str) -> bool:
        if "." not in path:
            return False
        splitted = path.split(".")
        return splitted[-2].endswith(base) and splitted[-1].lower() in [
            "unsmry",
            "funsmry",
        ]

    dir, base = os.path.split(case)
    candidates = list(filter(lambda x: _is_unsmry(base, x), os.listdir(dir or ".")))
    if not candidates:
        raise ValueError(f"Could not find any unsmry matching case path {case}")
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous reference to unsmry in {case}, could be any of {candidates}"
        )
    return os.path.join(dir, candidates[0])


class EclRun:
    """Wrapper class to run Eclipse simulations.

    The EclRun class is a small wrapper class which is used to run Eclipse
    simulations. It will load a configuration, i.e. where the binary is
    installed and so on, from an instance of the EclConfig class.

    The main method is the runEclipse() method which will:

      1. Set up redirection of the stdxxx file descriptors.
      2. Set the necessary environment variables.
      3. [MPI]: Create machine_file listing the nodes which should be used.
      4. fork+exec to actually run the Eclipse binary.
      5. Parse the output .PRT / .ECLEND file to check for errors.

    If the simulation fails the runEclipse() method will raise an exception.

    The class is called EclRun, and the main focus has been on running Eclipse
    simulations, but it should also handle "eclipse-like" simulators, e.g. the
    simulator OPM/flow.

    To actually create an executable script based on this class could in it's
    simplest form be:

       #!/usr/bin/env python
       import sys
       from .ecl_run import EclRun

       run = EclRun()
       run.runEclipse( )


    """

    def __init__(
        self,
        ecl_case: str,
        sim: Simulator,
        num_cpu: int = 1,
        check_status: bool = True,
        summary_conversion: bool = False,
    ):
        self.sim = sim
        self.check_status = check_status
        self.num_cpu = int(num_cpu)
        self.summary_conversion = summary_conversion

        # Dechipher the ecl_case argument.
        input_arg = ecl_case
        (_, ext) = os.path.splitext(input_arg)
        if ext and ext in [".data", ".DATA"]:
            data_file = input_arg
        elif input_arg.islower():
            data_file = input_arg + ".data"
        else:
            data_file = input_arg + ".DATA"

        if not os.path.isfile(data_file):
            raise IOError(f"No such file: {data_file}")

        (self.run_path, self.data_file) = os.path.split(data_file)
        (self.base_name, ext) = os.path.splitext(self.data_file)

        if self.run_path is None:
            self.run_path = os.getcwd()
        else:
            self.run_path = os.path.abspath(self.run_path)

    def runPath(self):
        return self.run_path

    def baseName(self):
        return self.base_name

    @property
    def prt_path(self):
        return Path(self.run_path) / (self.baseName() + ".PRT")

    def numCpu(self):
        return self.num_cpu

    def _get_legacy_run_env(self):
        my_env = os.environ.copy()
        my_env.update(self.sim.env.items())
        return my_env

    def initMPI(self):
        # If the environment variable LSB_MCPU_HOSTS is set we assume the job is
        # running on LSF - otherwise we assume it is running on the current host.
        #
        # If the LSB_MCPU_HOSTS variable is indeed set it will be a string like this:
        #
        #       host1 num_cpu1 host2 num_cpu2 ...
        #
        # i.e. an alternating list of hostname & number of
        # cpu. Alternatively/in addition the environment variable
        # LSB_HOSTS can be used. This variable is simply:
        #
        #       host1  host1  host2  host3

        LSB_MCPU_HOSTS = os.getenv("LSB_MCPU_HOSTS")
        LSB_HOSTS = os.getenv("LSB_HOSTS")

        if LSB_MCPU_HOSTS or LSB_HOSTS:
            LSB_MCPU_machine_list = make_LSB_MCPU_machine_list(LSB_MCPU_HOSTS)
            LSB_machine_list = make_LSB_machine_list(LSB_HOSTS)

            if len(LSB_MCPU_machine_list) == self.num_cpu:
                machine_list = LSB_MCPU_machine_list
            elif len(LSB_machine_list) == self.num_cpu:
                machine_list = LSB_machine_list
            else:
                raise EclError(
                    "LSF / MPI problems. "
                    f"Asked for:{self.num_cpu} cpu. "
                    f'LSB_MCPU_HOSTS: "{LSB_MCPU_HOSTS}"  LSB_HOSTS: "{LSB_HOSTS}"'
                )
        elif os.getenv("SLURM_JOB_NODELIST"):
            machine_list = make_SLURM_machine_list(
                os.getenv("SLURM_JOB_NODELIST"), os.getenv("SLURM_TASKS_PER_NODE")
            )
            if len(machine_list) != self.num_cpu:
                raise EclError(
                    f"SLURM / MPI problems - asked for {self.num_cpu} - "
                    f"got {len(machine_list)} nodes"
                )
        else:
            localhost = socket.gethostname()
            machine_list = [localhost] * self.num_cpu

        self.machine_file = f"{self.base_name}.mpi"
        with open(self.machine_file, "w", encoding="utf-8") as filehandle:
            for host in machine_list:
                filehandle.write(f"{host}\n")

    def _get_run_command(self, eclrun_config: EclrunConfig):
        summary_conversion = "yes" if self.summary_conversion else "no"
        return [
            "eclrun",
            "-v",
            eclrun_config.version,
            eclrun_config.simulator_name,
            f"{self.base_name}.DATA",
            "--summary-conversion",
            summary_conversion,
        ]

    def _get_legacy_run_command(self):
        if self.num_cpu == 1:
            return [self.sim.executable, self.base_name]
        else:
            self.initMPI()
            return [
                self.sim.mpirun,
                "-machinefile",
                self.machine_file,
                "-np",
                str(self.num_cpu),
                self.sim.executable,
                self.base_name,
            ]

    def _get_log_name(self, eclrun_config=None):
        # Eclipse version >= 2019.3 should log to a .OUT file
        # and not the legacy .LOG file.
        eclipse_version = (
            self.sim.version if eclrun_config is None else eclrun_config.version
        )

        logname_extension = "OUT"
        with suppress(version.InvalidVersion):
            if version.parse(eclipse_version) < version.parse("2019.3"):
                logname_extension = "LOG"
        return f"{self.base_name}.{logname_extension}"

    def execEclipse(self, eclrun_config=None) -> int:
        use_eclrun = eclrun_config is not None
        log_name = self._get_log_name(eclrun_config=eclrun_config)

        with pushd(self.run_path), open(log_name, "wb") as log_file:
            if not os.path.exists(self.data_file):
                raise IOError(f"Can not find data_file:{self.data_file}")
            if not os.access(self.data_file, os.R_OK):
                raise OSError(f"Can not read data file:{self.data_file}")

            command = (
                self._get_run_command(eclrun_config)
                if use_eclrun
                else self._get_legacy_run_command()
            )
            env = eclrun_config.run_env if use_eclrun else self._get_legacy_run_env()

            # await_process_tee() ensures the process is terminated.
            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
            )
            return await_process_tee(process, sys.stdout, log_file)

    LICENSE_FAILURE_RETRY_INITIAL_SLEEP = 90
    LICENSE_RETRY_STAGGER_FACTOR = 60
    LICENSE_RETRY_BACKOFF_EXPONENT = 3

    def runEclipse(self, eclrun_config=None, retries_left=3, backoff_sleep=None):
        backoff_sleep = (
            self.LICENSE_FAILURE_RETRY_INITIAL_SLEEP
            if backoff_sleep is None
            else backoff_sleep
        )
        return_code = self.execEclipse(eclrun_config=eclrun_config)

        OK_file = os.path.join(self.run_path, f"{self.base_name}.OK")
        if not self.check_status:
            with open(OK_file, "w", encoding="utf-8") as f:
                f.write("ECLIPSE simulation complete - NOT checked for errors.")
        else:
            if return_code != 0:
                command = (
                    self._get_run_command(eclrun_config)
                    if self.sim is None
                    else self._get_legacy_run_command()
                )
                raise subprocess.CalledProcessError(return_code, command)

            try:
                self.assertECLEND()
            except EclError as err:
                if err.failed_due_to_license_problems() and retries_left > 0:
                    time_to_wait = backoff_sleep + int(
                        random() * self.LICENSE_RETRY_STAGGER_FACTOR
                    )
                    sys.stderr.write(
                        "ECLIPSE failed due to license failure "
                        f"retrying in {time_to_wait} seconds\n"
                    )
                    time.sleep(time_to_wait)
                    self.runEclipse(
                        eclrun_config,
                        retries_left=retries_left - 1,
                        backoff_sleep=int(
                            backoff_sleep * self.LICENSE_RETRY_BACKOFF_EXPONENT
                        ),
                    )
                    return
                else:
                    raise err from None
            if self.num_cpu > 1:
                self.summary_block()

            with open(OK_file, "w", encoding="utf-8") as f:
                f.write("ECLIPSE simulation OK")

    def summary_block(self):
        case = os.path.join(self.run_path, self.base_name)
        start_time = datetime.datetime.now()
        prev_len = 0
        while True:
            dt = datetime.datetime.now() - start_time
            if dt.total_seconds() > 15:
                # We have not got a stable summary file after 15 seconds of
                # waiting, this either implies that something is completely
                # broken or this is a NOSIM simulation. Due the possibility of
                # NOSIM solution we just return here without signalling an
                # error.
                return None

            time.sleep(1)

            try:
                ecl_sum = [
                    r.read_keyword() for r in resfo.lazy_read(_find_unsmry(case))
                ]
            except Exception:
                continue

            this_len = len(ecl_sum)
            if prev_len == 0:
                prev_len = this_len
                continue

            if prev_len == this_len:
                break

        return ecl_sum

    def assertECLEND(self):
        tail_length = 5000
        result = self.readECLEND()
        if result.errors > 0:
            error_list = self.parseErrors()
            sep = "\n\n...\n\n"
            error_and_slave_msg = sep.join(error_list)
            extra_message = ""
            error_messages = [
                error for error in error_list if not "STARTING SLAVE" in str(error)
            ]
            if result.errors != len(error_messages):
                extra_message = (
                    f"\n\nWarning, mismatch between stated Error count ({result.errors}) "
                    f"and number of ERROR messages found in PRT ({len(error_messages)})."
                    f"\n\nTail ({tail_length} bytes) of PRT-file {self.prt_path}:\n\n"
                ) + tail_textfile(self.prt_path, 5000)

            raise EclError(
                "Eclipse simulation failed with:"
                f"{result.errors:d} errors:\n\n{error_and_slave_msg}{extra_message}"
            )

        if result.bugs > 0:
            raise EclError(f"Eclipse simulation failed with:{result.bugs:d} bugs")

    def readECLEND(self):
        error_regexp = re.compile(r"^\s*Errors\s+(\d+)\s*$")
        bug_regexp = re.compile(r"^\s*Bugs\s+(\d+)\s*$")

        report_file = os.path.join(self.run_path, f"{self.base_name}.ECLEND")
        if not os.path.isfile(report_file):
            report_file = self.prt_path

        errors = None
        bugs = None
        with open(report_file, "r", encoding="utf-8") as filehandle:
            for line in filehandle.readlines():
                error_match = re.match(error_regexp, line)
                if error_match:
                    errors = int(error_match.group(1))

                bug_match = re.match(bug_regexp, line)
                if bug_match:
                    bugs = int(bug_match.group(1))
        if errors is None:
            raise ValueError(f"Could not read errors from {report_file}")
        if bugs is None:
            raise ValueError(f"Could not read bugs from {report_file}")

        return EclipseResult(errors=errors, bugs=bugs)

    def parseErrors(self) -> List[str]:
        """Extract multiline ERROR messages from the PRT file"""
        error_list = []
        error_e100_regexp = re.compile(error_pattern_e100, re.MULTILINE)
        error_e300_regexp = re.compile(error_pattern_e300, re.MULTILINE)
        slave_started_regexp = re.compile(slave_started_pattern, re.MULTILINE)
        with open(self.prt_path, "r", encoding="utf-8") as filehandle:
            content = filehandle.read()

        for regexp in [error_e100_regexp, error_e300_regexp, slave_started_regexp]:
            offset = 0
            while True:
                match = regexp.search(content[offset:])
                if match:
                    error_list.append(
                        content[offset + match.start() : offset + match.end()]
                    )
                    offset += match.end()
                else:
                    break

        return error_list


def run(config: EclConfig, argv):
    parser = ArgumentParser()
    parser.add_argument("ecl_case")
    parser.add_argument("-v", "--version", dest="version", type=str)
    parser.add_argument("-n", "--num-cpu", dest="num_cpu", type=int, default=1)
    parser.add_argument(
        "-i", "--ignore-errors", dest="ignore_errors", action="store_true"
    )
    parser.add_argument(
        "--summary-conversion", dest="summary_conversion", action="store_true"
    )

    options = parser.parse_args(argv)

    try:
        eclrun_config = EclrunConfig(config, options.version)
        if eclrun_config.can_use_eclrun():
            run = EclRun(
                options.ecl_case,
                None,
                num_cpu=options.num_cpu,
                check_status=not options.ignore_errors,
                summary_conversion=options.summary_conversion,
            )
            run.runEclipse(eclrun_config=eclrun_config)
        else:
            if options.num_cpu > 1:
                sim = config.mpi_sim(version=options.version)
            else:
                sim = config.sim(version=options.version)

            run = EclRun(
                options.ecl_case,
                sim,
                num_cpu=options.num_cpu,
                check_status=not options.ignore_errors,
                summary_conversion=options.summary_conversion,
            )
            run.runEclipse()
    except EclError as msg:
        print(msg, file=sys.stderr)
        sys.exit(-1)


def tail_textfile(file_path: Path, num_chars: int) -> str:
    if not file_path.exists():
        return f"No output file {file_path}"
    with open(file_path, encoding="utf-8") as file:
        file.seek(0, 2)
        file_end_position = file.tell()
        seek_position = max(0, file_end_position - num_chars)
        file.seek(seek_position)
        return file.read()[-num_chars:]
