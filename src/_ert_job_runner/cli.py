import argparse
import json
import logging
import os
import signal
import sys
import typing
from datetime import datetime

from _ert_job_runner import reporting
from _ert_job_runner.reporting.message import Finish
from _ert_job_runner.runner import JobRunner

JOBS_FILE = "jobs.json"

logger = logging.getLogger(__name__)


def _setup_reporters(
    is_interactive_run,
    ens_id,
    dispatch_url,
    ee_token=None,
    ee_cert_path=None,
    experiment_id=None,
):
    reporters: typing.List[reporting.Reporter] = []
    if is_interactive_run:
        reporters.append(reporting.Interactive())
    elif ens_id and experiment_id is None:
        reporters.append(reporting.File())
        reporters.append(
            reporting.Event(
                evaluator_url=dispatch_url, token=ee_token, cert_path=ee_cert_path
            )
        )
    elif experiment_id:
        reporters.append(
            reporting.Protobuf(
                experiment_url=dispatch_url,
                token=ee_token,
                cert_path=ee_cert_path,
            )
        )
    else:
        reporters.append(reporting.File())
    return reporters


def _setup_logging(directory: str = "logs"):
    job_runner_logger = logging.getLogger("_ert_job_runner")
    os.makedirs(directory, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    filename = f"job-runner-log-{datetime.now().strftime('%Y-%m-%dT%H%M')}.txt"
    handler = logging.FileHandler(filename=directory + "/" + filename)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    job_runner_logger.addHandler(handler)
    job_runner_logger.setLevel(logging.DEBUG)


def main(args):

    parser = argparse.ArgumentParser(
        description=(
            "Run all the jobs specified in jobs.json, "
            "or specify the names of the jobs to run."
        )
    )
    parser.add_argument("run_path", nargs="?", help="Path where jobs.json is located")
    parser.add_argument(
        "job",
        nargs="*",
        help=(
            "One or more jobs to be executed from the jobs.json file. "
            "If no jobs are specified, all jobs will be executed."
        ),
    )

    parsed_args = parser.parse_args(args[1:])

    # If run_path is defined, enter into that directory
    if parsed_args.run_path is not None:
        if not os.path.exists(parsed_args.run_path):
            sys.exit(f"No such directory: {parsed_args.run_path}")
        os.chdir(parsed_args.run_path)

    # Make sure that logging is setup _after_ we have moved to the runpath directory
    _setup_logging()

    ens_id = None
    try:
        with open(JOBS_FILE, "r") as json_file:
            jobs_data = json.load(json_file)
            experiment_id = jobs_data.get("experiment_id")
            ens_id = jobs_data.get("ens_id")
            ee_token = jobs_data.get("ee_token")
            ee_cert_path = jobs_data.get("ee_cert_path")
            dispatch_url = jobs_data.get("dispatch_url")
    except ValueError as e:
        raise IOError(f"Job Runner cli failed to load JSON-file.{e}")

    is_interactive_run = len(parsed_args.job) > 0
    reporters = _setup_reporters(
        is_interactive_run,
        ens_id,
        dispatch_url,
        ee_token,
        ee_cert_path,
        experiment_id,
    )

    job_runner = JobRunner(jobs_data)

    for job_status in job_runner.run(parsed_args.job):
        logger.info(f"Job status: {job_status}")
        for reporter in reporters:
            reporter.report(job_status)

        if isinstance(job_status, Finish) and not job_status.success():
            pgid = os.getpgid(os.getpid())
            os.killpg(pgid, signal.SIGKILL)
