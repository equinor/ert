import functools
import json
import logging
import os
import socket
import time

from _ert_job_runner.io import cond_unlink
from _ert_job_runner.reporting.base import Reporter
from _ert_job_runner.reporting.message import (
    _JOB_EXIT_FAILED_STRING,
    _JOB_STATUS_FAILURE,
    _JOB_STATUS_RUNNING,
    _JOB_STATUS_SUCCESS,
    Exited,
    Finish,
    Init,
    Message,
    Running,
    Start,
)
from _ert_job_runner.util import data as data_util

TIME_FORMAT = "%H:%M:%S"
logger = logging.getLogger(__name__)
append = functools.partial(open, mode="a")

LOG_file = "JOB_LOG"
ERROR_file = "ERROR"
STATUS_file = "STATUS"
OK_file = "OK"
STATUS_json = "status.json"


class File(Reporter):
    def __init__(self):
        self.status_dict = {}
        self.node = socket.gethostname()

    def _update_status(self, job, **kwargs):
        self.status_dict["jobs"][job.index].update(**kwargs)

    def report(self, msg: Message):
        msg_ts = data_util.datetime_serialize(msg.timestamp)
        match msg:
            case Init(jobs, run_id):
                logger.debug("Init Message Instance")
                self._delete_old_status_files()
                self._init_status_file()
                self.status_dict = self._init_job_status_dict(msg_ts, run_id, jobs)
            case Start(job, error_message=None):
                logger.debug(f"Job {job.name()} was successfully started")
                self._start_status_file(msg)
                self._add_log_line(job)
                self._update_status(job, status=_JOB_STATUS_RUNNING, start_time=msg_ts)
            case Start(job, error_message=err):
                logger.error(f"Job {job.name()} FAILED to start")
                self._update_status(
                    job, status=_JOB_STATUS_FAILURE, error=err, end_time=msg_ts
                )
                self._complete_status_file(msg)

            case Exited(job, error_message=None):
                logger.debug(f"Job {job.name()} exited successfully")
                self._complete_status_file(msg)
                self._update_status(job, status=_JOB_STATUS_SUCCESS, end_time=msg_ts)
            case Exited(job, exit_code, error_message=err):
                logger.error(
                    _JOB_EXIT_FAILED_STRING.format(
                        job_name=job.name(),
                        exit_code=exit_code,
                        error_message=err,
                    )
                )
                self._update_status(
                    job, end_time=msg_ts, error=err, status=_JOB_STATUS_FAILURE
                )

                # A STATUS_file is not written if there is no exit_code, i.e.
                # when the job is killed due to timeout.
                if msg.exit_code:
                    self._complete_status_file(msg)
                self._dump_error_file(msg.job, err)
            case Running(job, max_memory_usage, current_memory_usage):
                self._update_status(
                    job,
                    max_memory_usage=max_memory_usage,
                    current_memory_usage=current_memory_usage,
                    status=_JOB_STATUS_RUNNING,
                )
            case Finish():
                logger.debug("Runner finished")
                if msg.success():
                    logger.debug("Runner finished successfully")
                    self.status_dict["end_time"] = msg_ts
                    self._dump_ok_file()
        self._dump_status_json()

    def _delete_old_status_files(self):
        logger.debug("Deleting old status files")
        cond_unlink(ERROR_file)
        cond_unlink(STATUS_file)
        cond_unlink(OK_file)

    def _write_status_file(self, msg: str) -> None:
        with append(file=STATUS_file) as status_file:
            status_file.write(msg)

    def _init_status_file(self):
        self._write_status_file(f"{'Current host':32}: {self.node}/{os.uname()[4]}\n")

    @staticmethod
    def _init_job_status_dict(start_time, run_id, jobs):
        return {
            "run_id": run_id,
            "start_time": start_time,
            "end_time": None,
            "jobs": [data_util.create_job_dict(j) for j in jobs],
        }

    def _start_status_file(self, msg):
        timestamp = msg.timestamp.strftime(TIME_FORMAT)
        job_name = msg.job.name()
        self._write_status_file(f"{job_name:32}: {timestamp} .... ")
        logger.info(
            f"Append {job_name} job starting timestamp {timestamp} to STATUS_file."
        )

    def _complete_status_file(self, msg):
        status: str = ""
        timestamp = msg.timestamp.strftime(TIME_FORMAT)
        if not msg.success():
            # There was no status code in the case of STARTUP_ERROR, so use
            # an arbitrary code less than -9.
            exit_code = -10 if isinstance(msg, Start) else msg.exit_code
            status = f" EXIT: {exit_code}/{msg.error_message}"
            logger.error(f"{msg.job.name()} job, {timestamp} {status}")
        self._write_status_file(f"{timestamp}  {status}\n")

    def _add_log_line(self, job):
        with append(file=LOG_file) as f:
            args = " ".join(job.job_data["argList"])
            time_str = time.strftime(TIME_FORMAT, time.localtime())
            f.write(f"{time_str}  Calling: {job.job_data['executable']} {args}\n")

    # This file will be read by the job_queue_node_fscanf_EXIT() function
    # in job_queue.c. Be very careful with changes in output format.
    def _dump_error_file(self, job, error_msg):
        with append(ERROR_file) as file:
            file.write("<error>\n")
            file.write(
                f"  <time>{time.strftime(TIME_FORMAT, time.localtime())}</time>\n"
            )
            file.write(f"  <job>{job.name()}</job>\n")
            file.write(f"  <reason>{error_msg}</reason>\n")
            stderr_file = None
            if job.std_err:
                if os.path.exists(job.std_err):
                    with open(job.std_err, "r", encoding="utf-8") as error_file_handler:
                        stderr = error_file_handler.read()
                        if stderr:
                            stderr_file = os.path.join(os.getcwd(), job.std_err)
                        else:
                            stderr = f"<Not written by:{job.name()}>\n"
                else:
                    stderr = f"<stderr: Could not find file:{job.std_err}>\n"
            else:
                stderr = "<stderr: Not redirected>\n"

            file.write(f"  <stderr>\n{stderr}</stderr>\n")
            if stderr_file:
                file.write(f"  <stderr_file>{stderr_file}</stderr_file>\n")

            file.write("</error>\n")

    def _dump_ok_file(self):
        with open(OK_file, "w", encoding="utf-8") as f:
            f.write(
                f"All jobs complete {time.strftime(TIME_FORMAT, time.localtime())} \n"
            )

    def _dump_status_json(self):
        with open(STATUS_json, "w", encoding="utf-8") as fp:
            json.dump(self.status_dict, fp, indent=4)
