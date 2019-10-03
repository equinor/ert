import os
import signal
import sys

import job_runner.reporting as reporting
from job_runner.reporting.message import Finish
from job_runner.runner import JobRunner
from job_runner.util import check_version


def main(args):
    # If run_path is defined, enter into that directory
    if len(args) > 1:
        run_path = args[1]

        if not os.path.exists(run_path):
            sys.exit("No such directory: {}".format(run_path))
        os.chdir(run_path)

    warning = check_version()
    if warning is not None:
        sys.stderr.write(warning)

    jobs_to_run = []
    if len(args) > 2:
        jobs_to_run = args[2:]

    reporters = []
    is_interactive_run = len(args) > 2
    if is_interactive_run:
        reporters.append(reporting.Interactive())
    else:
        reporters.append(reporting.File())
        reporters.append(reporting.Network())

    job_runner = JobRunner()

    for job_status in job_runner.run(jobs_to_run):
        for reporter in reporters:
            reporter.report(job_status)

        if isinstance(job_status, Finish) and not job_status.success():
            pgid = os.getpgid(os.getpid())
            os.killpg(pgid, signal.SIGKILL)
