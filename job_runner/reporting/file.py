import json
import os
import socket
import time

from job_runner.io import cond_unlink
from job_runner.reporting.message import (
    _JOB_STATUS_FAILURE,
    _JOB_STATUS_RUNNING,
    _JOB_STATUS_SUCCESS,
    Exited,
    Finish,
    Init,
    Running,
    Start,
)
from job_runner.reporting.base import Reporter
from job_runner.util import data as data_util

TIME_FORMAT = "%H:%M:%S"


class File(Reporter):
    LOG_file = "JOB_LOG"
    ERROR_file = "ERROR"
    STATUS_file = "STATUS"
    OK_file = "OK"
    STATUS_json = "status.json"

    def __init__(self, sync_disc_timeout=10):
        self.status_dict = {}
        self.node = socket.gethostname()
        self._sync_disc_timeout = sync_disc_timeout

    def report(self, msg):
        job_status = {}
        if msg.job:
            index = msg.job.index
            job_status = self.status_dict["jobs"][index]

        if isinstance(msg, Init):
            self._delete_old_status_files()
            self._init_status_file()
            self.status_dict = self._init_job_status_dict(
                msg.timestamp, msg.run_id, msg.jobs
            )

        elif isinstance(msg, Start):
            if msg.success():
                self._start_status_file(msg)
                self._add_log_line(msg.job)
                job_status["status"] = _JOB_STATUS_RUNNING
                job_status["start_time"] = data_util.datetime_serialize(msg.timestamp)
            else:
                error_msg = msg.error_message
                job_status["status"] = _JOB_STATUS_FAILURE
                job_status["error"] = error_msg
                job_status["end_time"] = data_util.datetime_serialize(msg.timestamp)

                self._complete_status_file(msg)
        elif isinstance(msg, Exited):
            job_status["end_time"] = data_util.datetime_serialize(msg.timestamp)

            if msg.success():
                job_status["status"] = _JOB_STATUS_SUCCESS
                self._complete_status_file(msg)
            else:
                error_msg = msg.error_message
                job_status["error"] = error_msg
                job_status["status"] = _JOB_STATUS_FAILURE

                # A STATUS_file is not written if there is no exit_code, i.e.
                # when the job is killed due to timeout.
                if msg.exit_code:
                    self._complete_status_file(msg)
                self._dump_error_file(msg.job, error_msg)

        elif isinstance(msg, Running):
            job_status["max_memory_usage"] = msg.max_memory_usage
            job_status["current_memory_usage"] = msg.current_memory_usage
            job_status["status"] = _JOB_STATUS_RUNNING

        elif isinstance(msg, Finish):
            if msg.success():
                self.status_dict["end_time"] = data_util.datetime_serialize(
                    msg.timestamp
                )
                self._dump_ok_file()
        self._dump_status_json()

    def _delete_old_status_files(self):
        cond_unlink(self.ERROR_file)
        cond_unlink(self.STATUS_file)
        cond_unlink(self.OK_file)

    def _init_status_file(self):
        with open(self.STATUS_file, "a") as f:
            f.write("{:32}: {}/{}\n".format("Current host", self.node, os.uname()[4]))

    def _init_job_status_dict(self, start_time, run_id, jobs):
        return {
            "run_id": run_id,
            "start_time": data_util.datetime_serialize(start_time),
            "end_time": None,
            "jobs": [data_util.create_job_dict(j) for j in jobs],
        }

    def _start_status_file(self, msg):
        with open(self.STATUS_file, "a") as f:
            f.write(f"{msg.job.name():32}: {msg.timestamp.strftime(TIME_FORMAT)} .... ")

    def _complete_status_file(self, msg):
        status: str = ""
        if not msg.success():
            # There was no status code in the case of STARTUP_ERROR, so use
            # an arbitrary code less than -9.
            exit_code = -10 if isinstance(msg, Start) else msg.exit_code
            status = f" EXIT: {exit_code}/{msg.error_message}"
        with open(self.STATUS_file, "a") as f:
            f.write(f"{msg.timestamp.strftime(TIME_FORMAT)}  {status}\n")

    def _add_log_line(self, job):
        with open(self.LOG_file, "a") as f:
            args = " ".join(job.job_data["argList"])
            time_str = time.strftime(TIME_FORMAT, time.localtime())
            f.write(f"{time_str}  Calling: {job.job_data['executable']} {args}\n")

    # This file will be read by the job_queue_node_fscanf_EXIT() function
    # in job_queue.c. Be very careful with changes in output format.
    def _dump_error_file(self, job, error_msg):
        with open(self.ERROR_file, "a") as file:
            file.write("<error>\n")
            file.write(
                f"  <time>{time.strftime(TIME_FORMAT, time.localtime())}</time>\n"
            )
            file.write(f"  <job>{job.name()}</job>\n")
            file.write(f"  <reason>{error_msg}</reason>\n")
            stderr_file = None
            if job.std_err:
                if os.path.exists(job.std_err):
                    with open(job.std_err, "r") as error_file_handler:
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
        with open(self.OK_file, "w") as f:
            f.write(
                f"All jobs complete {time.strftime(TIME_FORMAT, time.localtime())} \n"
            )
        time.sleep(self._sync_disc_timeout)  # Let the disks sync up

    def _dump_status_json(self):
        with open(self.STATUS_json, "w") as fp:
            json.dump(self.status_dict, fp, indent=1)
