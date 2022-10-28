import datetime
import os
import os.path
import re
import socket
import subprocess
import sys
import time
from argparse import ArgumentParser
from collections import namedtuple
from contextlib import contextmanager

from ecl.summary import EclSum
from ecl_config import EclrunConfig
from packaging import version


def await_process_tee(process, *out_files):
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
error_pattern = rf"^\s@--  ERROR{date_sub_pattern}${body_sub_pattern}"


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
        if mult_string is None:
            mult = 1
        else:
            mult = int(mult_string)

        return [count] * mult
    else:
        raise Exception(f"Failed to parse SLURM_TASKS_PER_NODE: {task_count_string}")


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
       from ert._c_wrappers.fm.ecl import EclRun

       run = EclRun()
       run.runEclipse( )


    """

    def __init__(
        self, ecl_case, sim, num_cpu=1, check_status=True, summary_conversion=False
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
        else:
            if input_arg.islower():
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
                raise Exception(
                    "LSF / MPI problems. "
                    f"Asked for:{self.num_cpu} cpu. "
                    f'LSB_MCPU_HOSTS: "{LSB_MCPU_HOSTS}"  LSB_HOSTS: "{LSB_HOSTS}"'
                )
        elif os.getenv("SLURM_JOB_NODELIST"):
            machine_list = make_SLURM_machine_list(
                os.getenv("SLURM_JOB_NODELIST"), os.getenv("SLURM_TASKS_PER_NODE")
            )
            if len(machine_list) != self.num_cpu:
                raise Exception(
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

        return (
            f"{self.base_name}.OUT"
            if version.parse(eclipse_version) >= version.parse("2019.3")
            else f"{self.base_name}.LOG"
        )

    def execEclipse(self, eclrun_config=None):
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

            # pylint: disable=consider-using-with
            # await_process_tee() ensures the process is terminated.
            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
            )
            return await_process_tee(process, sys.stdout, log_file)

    def runEclipse(self, eclrun_config=None):
        return_code = self.execEclipse(eclrun_config=eclrun_config)

        OK_file = os.path.join(self.run_path, f"{self.base_name}.OK")
        if not self.check_status:
            with open(OK_file, "w", encoding="utf-8") as f:
                f.write("ECLIPSE simulation complete - NOT checked for errors.")
        else:
            if return_code != 0:
                raise Exception(
                    f"The eclipse executable exited with error status: {return_code:d}"
                )

            self.assertECLEND()
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
                ecl_sum = EclSum(case)
            except (OSError, ValueError):
                continue

            this_len = len(ecl_sum)
            if prev_len == 0:
                prev_len = this_len
                continue

            if prev_len == this_len:
                break

        return ecl_sum

    def assertECLEND(self):
        result = self.readECLEND()
        if result.errors > 0:
            error_list = self.parseErrors()
            sep = "\n\n...\n\n"
            error_msg = sep.join(error_list)
            raise Exception(
                "Eclipse simulation failed with:"
                f"{result.errors:d} errors:\n\n{error_msg}"
            )

        if result.bugs > 0:
            raise Exception(f"Eclipse simulation failed with:{result.bugs:d} bugs")

    def readECLEND(self):
        error_regexp = re.compile(r"^\s*Errors\s+(\d+)\s*$")
        bug_regexp = re.compile(r"^\s*Bugs\s+(\d+)\s*$")

        report_file = os.path.join(self.run_path, f"{self.base_name}.ECLEND")
        if not os.path.isfile(report_file):
            report_file = os.path.join(self.run_path, f"{self.base_name}.PRT")

        with open(report_file, "r", encoding="utf-8") as filehandle:
            for line in filehandle.readlines():
                error_match = re.match(error_regexp, line)
                if error_match:
                    errors = int(error_match.group(1))

                bug_match = re.match(bug_regexp, line)
                if bug_match:
                    bugs = int(bug_match.group(1))

        return EclipseResult(errors=errors, bugs=bugs)

    def parseErrors(self):
        prt_file = os.path.join(self.runPath(), f"{self.baseName()}.PRT")
        error_list = []
        error_regexp = re.compile(error_pattern, re.MULTILINE)
        with open(prt_file, "r", encoding="utf-8") as filehandle:
            content = filehandle.read()

        offset = 0
        while True:
            match = error_regexp.search(content[offset:])
            if match:
                error_list.append(
                    content[offset + match.start() : offset + match.end()]
                )
                offset += match.end()
            else:
                break

        return error_list


def run(config, argv):
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
