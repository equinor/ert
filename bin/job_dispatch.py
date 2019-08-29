#!/usr/bin/env python

import os
import sys
import signal

import job_runner.reporting as reporting
from job_runner.reporting.message import Exited
from job_runner.runner import JobRunner
from job_runner.util import check_version

LOG_URL = "http://devnull.statoil.no:4444"


def main():
    # If run_path is defined, enter into that directory
    if len(sys.argv) > 1:
        run_path = sys.argv[1]

        if not os.path.exists(run_path):
            sys.stderr.write("No such directory: %s\n" % run_path)
            sys.exit(1)
        os.chdir(run_path)

    warning = check_version()
    if warning is not None:
        sys.stderr.write(warning)

    jobs_to_run = []
    if len(sys.argv) > 2:
        jobs_to_run = sys.argv[2:]

    reporters = []
    is_interactive_run = len(sys.argv) > 2
    if is_interactive_run:
        reporters.append(reporting.Interactive())
    else:
        reporters.append(reporting.File())
        reporters.append(reporting.Network(error_url=LOG_URL))

    job_runner = JobRunner()

    for job_status in job_runner.run(jobs_to_run):
        for reporter in reporters:
            reporter.report(job_status)

        if isinstance(job_status, Exited) and not job_status.success():
            pgid = os.getpgid(os.getpid())
            os.killpg(pgid, signal.SIGKILL)


def sigterm_handler(_signo, _stack_frame):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(0, signal.SIGTERM)
#################################################################


#################################################################
os.nice(19)
if __name__ == "__main__":
    signal.signal(signal.SIGTERM, sigterm_handler)
    main()
