import argparse
import contextlib
import json
import logging
import os
import signal
import sys
import time
from collections.abc import Generator, Iterable
from datetime import datetime

from _ert.forward_model_runner import reporting
from _ert.forward_model_runner.reporting.message import (
    Exited,
    Finish,
    Message,
    ProcessTreeStatus,
)
from _ert.forward_model_runner.runner import ForwardModelRunner

# This is incorrecty named, but is kept to avoid a breaking change.
# "job" was previously used for what is now called a "forward_model_step".
FORWARD_MODEL_DESCRIPTION_FILE = "jobs.json"

# On shared filesystems race conditions between different computers can occur
# yielding FileNotFoundError due to synchronization issues. This constant
# determines how long we can wait for synchronization to happen:
FILE_RETRY_TIME = 30
FORWARD_MODEL_TERMINATED_MSG = "Forward model was terminated"

logger = logging.getLogger(__name__)


def _setup_reporters(
    is_interactive_run,
    ens_id,
    dispatch_url,
    ee_token=None,
    experiment_id=None,
) -> list[reporting.Reporter]:
    reporters: list[reporting.Reporter] = []
    if is_interactive_run:
        reporters.append(reporting.Interactive())
    elif ens_id and experiment_id is None:
        reporters.append(reporting.File())
        if dispatch_url is not None:
            reporters.append(
                reporting.Event(evaluator_url=dispatch_url, token=ee_token)
            )
    else:
        reporters.append(reporting.File())
    return reporters


def _setup_logging(directory: str = "logs"):
    fm_runner_logger = logging.getLogger("_ert.forward_model_runner")
    memory_csv_logger = logging.getLogger("_ert.forward_model_memory_profiler")

    os.makedirs(directory, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    filename = (
        f"forward-model-runner-log-{datetime.now().strftime('%Y-%m-%dT%H%M')}.txt"
    )
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

    fm_runner_logger.addHandler(handler)
    fm_runner_logger.setLevel(logging.DEBUG)


def _wait_for_retry():
    time.sleep(FILE_RETRY_TIME)


def _read_fm_description_file(retry=True):
    try:
        with open(FORWARD_MODEL_DESCRIPTION_FILE, encoding="utf-8") as json_file:
            return json.load(json_file)
    except json.JSONDecodeError as e:
        raise OSError(
            "fm_dispatch failed to load JSON-file describing the forward model."
        ) from e
    except FileNotFoundError as e:
        if retry:
            logger.error(
                f"Could not find file {FORWARD_MODEL_DESCRIPTION_FILE}, retrying"
            )
            _wait_for_retry()
            return _read_fm_description_file(retry=False)
        else:
            raise e


def _report_all_messages(
    messages: Generator[Message],
    reporters: list[reporting.Reporter],
) -> None:
    for msg in messages:
        logger.info(f"Forward model status: {msg}")
        i = 0
        while i < len(reporters):
            reporter = reporters[i]
            try:
                reporter.report(msg)
                i += 1
            except Exception as err:
                with contextlib.suppress(Exception):
                    del reporters[i]
                    if isinstance(reporter, reporting.Event):
                        reporter.stop()
                    logger.exception(
                        f"Reporter {reporter} failed due to {err}."
                        " Removing the reporter."
                    )
        if isinstance(msg, Finish) and not msg.success():
            _stop_reporters_and_sigkill(reporters)


def _stop_reporters_and_sigkill(reporters, exited_event: Exited | None = None):
    _stop_reporters(reporters, exited_event)
    pgid = os.getpgid(os.getpid())
    os.killpg(pgid, signal.SIGKILL)


def _stop_reporters(
    reporters: Iterable[reporting.Reporter], exited_event: Exited | None = None
) -> None:
    for reporter in reporters:
        if isinstance(reporter, reporting.Event):
            reporter.stop(exited_event=exited_event)


def sigterm_handler(_signo, _stack_frame):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(0, signal.SIGTERM)


def fm_dispatch(args):
    parser = argparse.ArgumentParser(
        description=(
            "Run all the forward model steps specified in jobs.json, "
            "or specify the names of the steps to run."
        )
    )
    parser.add_argument("run_path", nargs="?", help="Path where jobs.json is located")
    parser.add_argument(
        "steps",
        nargs="*",
        help=(
            "One or more forward model steps to be executed from the jobs.json file. "
            "If no steps are specified, all steps will be executed."
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

    fm_description = _read_fm_description_file()

    experiment_id = fm_description.get("experiment_id")
    ens_id = fm_description.get("ens_id")
    ee_token = fm_description.get("ee_token")
    dispatch_url = fm_description.get("dispatch_url")

    is_interactive_run = len(parsed_args.steps) > 0
    reporters = _setup_reporters(
        is_interactive_run,
        ens_id,
        dispatch_url,
        ee_token,
        experiment_id,
    )

    fm_runner = ForwardModelRunner(fm_description)

    def sigterm_handler(_signo, _stack_frame):
        exited_event = Exited(
            fm_runner._currently_running_step, exit_code=1
        ).with_error(FORWARD_MODEL_TERMINATED_MSG)
        _stop_reporters_and_sigkill(reporters, exited_event)

    signal.signal(signal.SIGTERM, sigterm_handler)
    _report_all_messages(fm_runner.run(parsed_args.steps), reporters)


def main():
    os.nice(19)
    signal.signal(signal.SIGTERM, sigterm_handler)
    try:
        fm_dispatch(sys.argv)
    except Exception as e:
        pgid = os.getpgid(os.getpid())
        os.killpg(pgid, signal.SIGTERM)
        raise e


if __name__ == "__main__":
    main()
