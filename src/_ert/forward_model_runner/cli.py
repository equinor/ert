import argparse
import json
import logging
import os
import signal
import sys
import time
import typing
from datetime import datetime

from _ert.forward_model_runner import reporting
from _ert.forward_model_runner.reporting.message import Finish, ProcessTreeStatus
from _ert.forward_model_runner.runner import ForwardModelRunner

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
    else:
        reporters.append(reporting.File())
    return reporters


def _setup_logging(directory: str = "logs"):
    job_runner_logger = logging.getLogger("_ert.forward_model_runner")
    memory_csv_logger = logging.getLogger("_ert.forward_model_memory_profiler")

    os.makedirs(directory, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    filename = f"job-runner-log-{datetime.now().strftime('%Y-%m-%dT%H%M')}.txt"
    csv_filename = f"memory-profile-{datetime.now().strftime('%Y-%m-%dT%H%M')}.csv"

    handler = logging.FileHandler(filename=directory + "/" + filename)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    csv_handler = logging.FileHandler(filename=directory + "/" + csv_filename)
    csv_handler.setFormatter(logging.Formatter("%(message)s"))
    memory_csv_logger.addHandler(csv_handler)
    memory_csv_logger.setLevel(logging.INFO)
    # Write the CSV header to the file:
    memory_csv_logger.info(ProcessTreeStatus().csv_header())

    job_runner_logger.addHandler(handler)
    job_runner_logger.setLevel(logging.DEBUG)


JOBS_JSON_RETRY_TIME = 30


def _wait_for_retry():
    time.sleep(JOBS_JSON_RETRY_TIME)


def _read_jobs_file(retry=True):
    try:
        with open(JOBS_FILE, "r", encoding="utf-8") as json_file:
            return json.load(json_file)
    except json.JSONDecodeError as e:
        raise IOError("Job Runner cli failed to load JSON-file.") from e
    except FileNotFoundError as e:
        if retry:
            logger.error(f"Could not find file {JOBS_FILE}, retrying")
            _wait_for_retry()
            return _read_jobs_file(retry=False)
        else:
            raise e


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

    jobs_data = _read_jobs_file()

    experiment_id = jobs_data.get("experiment_id")
    ens_id = jobs_data.get("ens_id")
    ee_token = jobs_data.get("ee_token")
    ee_cert_path = jobs_data.get("ee_cert_path")
    dispatch_url = jobs_data.get("dispatch_url")

    is_interactive_run = len(parsed_args.job) > 0
    reporters = _setup_reporters(
        is_interactive_run,
        ens_id,
        dispatch_url,
        ee_token,
        ee_cert_path,
        experiment_id,
    )

    job_runner = ForwardModelRunner(jobs_data)

    for job_status in job_runner.run(parsed_args.job):
        logger.info(f"Job status: {job_status}")
        for reporter in reporters:
            try:
                reporter.report(job_status)
            except OSError as oserror:
                print(
                    f"job_dispatch failed due to {oserror}. Stopping and cleaning up."
                )
                pgid = os.getpgid(os.getpid())
                os.killpg(pgid, signal.SIGKILL)

        if isinstance(job_status, Finish) and not job_status.success():
            pgid = os.getpgid(os.getpid())
            os.killpg(pgid, signal.SIGKILL)
