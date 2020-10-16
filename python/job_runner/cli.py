import argparse
import os
import signal
import sys

import job_runner.reporting as reporting
from job_runner.reporting.message import Finish
from job_runner.runner import JobRunner


def main(args):

    parser = argparse.ArgumentParser(
        description="Run all the jobs specified in jobs.json, or specify the names of the jobs to run."
    )
    parser.add_argument("run_path", nargs="?", help="Path where jobs.json is located")
    parser.add_argument(
        "job",
        nargs="*",
        help="One or more jobs to be executed from the jobs.json file. If no jobs are specified, all jobs will be executed.",
    )

    parsed_args = parser.parse_args(args[1:])

    # If run_path is defined, enter into that directory
    if parsed_args.run_path is not None:
        if not os.path.exists(parsed_args.run_path):
            sys.exit("No such directory: {}".format(parsed_args.run_path))
        os.chdir(parsed_args.run_path)

    reporters = []

    if len(parsed_args.job) > 0:
        reporters.append(reporting.Interactive())
    else:
        reporters.append(reporting.File())
        reporters.append(reporting.Network())

    job_runner = JobRunner()

    for job_status in job_runner.run(parsed_args.job):
        for reporter in reporters:
            reporter.report(job_status)

        if isinstance(job_status, Finish) and not job_status.success():
            pgid = os.getpgid(os.getpid())
            os.killpg(pgid, signal.SIGKILL)
