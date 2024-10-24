#!/usr/bin/env python
import datetime
import glob
import os
import os.path
import re
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path
from random import random
from typing import List, Literal, Optional

import resfo


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


def find_unsmry(basepath: Path) -> Path:
    def _is_unsmry(base: str, path: str) -> bool:
        if "." not in path:
            return False
        splitted = path.split(".")
        return splitted[-2].endswith(base) and splitted[-1].lower() in [
            "unsmry",
            "funsmry",
        ]

    dir = basepath.parent
    base = basepath.name
    candidates: List[str] = list(
        filter(lambda x: _is_unsmry(base, x), os.listdir(dir or "."))
    )
    if not candidates:
        raise ValueError(f"Could not find any unsmry matching case path {basepath}")
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous reference to unsmry in {basepath}, could be any of {candidates}"
        )
    return Path(dir) / candidates[0]


def await_completed_unsmry_file(
    smry_path: Path, max_wait: float = 15, poll_interval: float = 1.0
) -> float:
    """This function will wait until the provided smry file does not grow in size
    during one poll interval.

    Such a wait is sometimes needed when different MPI hosts write data to a shared
    disk system.

    If the file does not exist or is completely unreadable to resfo, this function
    will timeout to max_wait. If NOSIM is included, this will happen.

    Size is defined in terms of readable data elementes through resfo.

    This function will always wait for at least one poll interval, the polling
    interval is specified in seconds.

    The return value is the waited time (in seconds)"""
    start_time = datetime.datetime.now()
    prev_len = 0
    while (datetime.datetime.now() - start_time).total_seconds() < max_wait:
        try:
            resfo_sum = [r.read_keyword() for r in resfo.lazy_read(smry_path)]
        except Exception:
            time.sleep(poll_interval)
            continue

        current_len = len(resfo_sum)
        if prev_len == current_len:
            # smry file is regarded complete
            break
        else:
            prev_len = max(prev_len, current_len)

        time.sleep(poll_interval)

    return (datetime.datetime.now() - start_time).total_seconds()


class RunReservoirSimulator:
    """Wrapper class to run system installed `eclrun` or `flowrun`.

    Will initiate a limited number of reruns if license errors are detected, also
    in coupled simulations (not relevant for Flow)

    PRT/ECLEND files are checked for errors, and exceptions will be raised.
    """

    def __init__(
        self,
        simulator: Literal["flow", "eclipse", "e300"],
        version: str,
        ecl_case: str,  # consider Path
        num_cpu: int = 1,
        check_status: bool = True,
        summary_conversion: bool = False,
    ):
        self.simulator = simulator
        self.version: str = version

        self.num_cpu: int = int(num_cpu)
        self.check_status: bool = check_status
        self.summary_conversion: bool = summary_conversion

        self.bypass_flowrun: bool = False

        _runner_abspath: Optional[str] = None
        if simulator in ["eclipse", "e300"]:
            _runner_abspath: Optional[str] = shutil.which("eclrun")
            if _runner_abspath is None:
                raise RuntimeError("eclrun not installed")
        else:
            _runner_abspath: Optional[str] = shutil.which("flowrun")
            if _runner_abspath is None:
                _runner_abspath = shutil.which("flow")
                if _runner_abspath is None:
                    raise RuntimeError("flowrun or flow not installed")
                else:
                    if self.num_cpu > 1:
                        raise RuntimeError(
                            "MPI runs not supported without a flowrun wrapper"
                        )
                    self.bypass_flowrun = True
        self.runner_abspath: str = _runner_abspath

        data_file = ecl_case_to_data_file(ecl_case)

        if not Path(data_file).exists:
            raise IOError(f"No such file: {data_file}")

        self.run_path: Path = Path(data_file).parent.absolute()
        self.data_file: str = Path(data_file).name
        self.base_name: str = Path(data_file).stem

    @property
    def prt_path(self) -> Path:
        return self.run_path / (self.base_name + ".PRT")

    @property
    def eclrun_command(self) -> List[str]:
        return [
            self.runner_abspath,
            self.simulator,
            "--version",
            self.version,
            self.data_file,
            "--summary-conversion",
            "yes" if self.summary_conversion else "no",
            # "-np", self.num_cpu  # eclrun detects automatically
        ]

    @property
    def flowrun_command(self) -> List[str]:
        if self.bypass_flowrun:
            return [
                self.runner_abspath,
                self.data_file,
            ]
        return [
            self.runner_abspath,
            "--version",
            self.version,
            self.data_file,
            "--np",
            str(self.num_cpu),
        ]

    def runFlow(self) -> None:
        return_code = subprocess.run(self.flowrun_command, check=False).returncode
        OK_file = self.run_path / f"{self.base_name}.OK"
        if not self.check_status:
            OK_file.write_text(
                "FLOW simulation complete - NOT checked for errors.",
                encoding="utf-8",
            )
        else:
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, self.flowrun_command)
            self.assertECLEND()
            if self.num_cpu > 1:
                await_completed_unsmry_file(find_unsmry(self.run_path / self.base_name))

            OK_file.write_text("FLOW simulation OK", encoding="utf-8")

    LICENSE_FAILURE_RETRY_INITIAL_SLEEP = 90
    LICENSE_RETRY_STAGGER_FACTOR = 60
    LICENSE_RETRY_BACKOFF_EXPONENT = 3

    def runEclipseX00(self, retries_left=3, backoff_sleep=None) -> None:
        # This function calls itself recursively in case of license failures
        backoff_sleep = (
            self.LICENSE_FAILURE_RETRY_INITIAL_SLEEP
            if backoff_sleep is None
            else backoff_sleep
        )
        return_code = subprocess.run(self.eclrun_command, check=False).returncode

        OK_file = self.run_path / f"{self.base_name}.OK"
        if not self.check_status:
            OK_file.write_text(
                "ECLIPSE simulation complete - NOT checked for errors.",
                encoding="utf-8",
            )
        else:
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, self.eclrun_command)
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
                    self.runEclipseX00(
                        retries_left=retries_left - 1,
                        backoff_sleep=int(
                            backoff_sleep * self.LICENSE_RETRY_BACKOFF_EXPONENT
                        ),
                    )
                    return
                else:
                    raise err from None
            if self.num_cpu > 1:
                await_completed_unsmry_file(find_unsmry(self.run_path / self.base_name))

            OK_file.write_text("ECLIPSE simulation OK", encoding="utf-8")

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


def tail_textfile(file_path: Path, num_chars: int) -> str:
    if not file_path.exists():
        return f"No output file {file_path}"
    with open(file_path, encoding="utf-8") as file:
        file.seek(0, 2)
        file_end_position = file.tell()
        seek_position = max(0, file_end_position - num_chars)
        file.seek(seek_position)
        return file.read()[-num_chars:]


def run_reservoirsimulator(args: List[str]):
    parser = ArgumentParser()
    parser.add_argument("simulator", type=str, choices=["flow", "eclipse", "e300"])
    parser.add_argument("version", type=str)
    parser.add_argument("ecl_case", type=str)
    parser.add_argument("-n", "--num-cpu", dest="num_cpu", type=int, default=1)
    parser.add_argument(
        "-i", "--ignore-errors", dest="ignore_errors", action="store_true"
    )
    parser.add_argument(
        "--summary-conversion", dest="summary_conversion", action="store_true"
    )

    options = parser.parse_args(args)

    if options.summary_conversion and options.simulator == "flow":
        # or is this the esmry option to flow?
        raise RuntimeError("--summary-conversion is not available with simulator flow")

    try:
        if options.simulator in ["eclipse", "e300"]:
            RunReservoirSimulator(
                options.simulator,
                options.version,
                options.ecl_case,
                num_cpu=options.num_cpu,
                check_status=not options.ignore_errors,
                summary_conversion=options.summary_conversion,
            ).runEclipseX00()
        else:
            RunReservoirSimulator(
                "flow",
                options.version,
                options.ecl_case,
                num_cpu=options.num_cpu,
                check_status=not options.ignore_errors,
            ).runFlow()

    except EclError as msg:
        print(msg, file=sys.stderr)
        sys.exit(-1)


def ecl_case_to_data_file(ecl_case: str) -> str:
    ext: str = Path(ecl_case).suffix
    if ext in [".data", ".DATA"]:
        return ecl_case
    elif ecl_case.islower():
        return ecl_case + ".data"
    else:
        return ecl_case + ".DATA"


if __name__ == "__main__":
    non_empty_args = list(filter(None, sys.argv))
    run_reservoirsimulator(non_empty_args[1:])
