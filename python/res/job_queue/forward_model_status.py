#  Copyright (C) 2018  Equinor ASA, Norway.
#
#  The file 'forward_model_status.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
import os.path
import json
import datetime
import time
import sys
from job_runner.reporting.file import File
from job_runner import JOBS_FILE


def _serialize_date(dt):
    if dt is None:
        return None

    return time.mktime(dt.timetuple())


def _deserialize_date(serial_dt):
    if serial_dt is None:
        return None

    time_struct = time.localtime(serial_dt)
    return datetime.datetime(*time_struct[0:6])


class ForwardModelJobStatus(object):
    def __init__(
        self,
        name,
        start_time=None,
        end_time=None,
        status="Waiting",
        error=None,
        std_out_file="",
        std_err_file="",
        current_memory_usage=0,
        max_memory_usage=0,
    ):

        self.start_time = start_time
        self.end_time = end_time
        self.name = name
        self.status = status
        self.error = error
        self.std_out_file = std_out_file
        self.std_err_file = std_err_file
        self.current_memory_usage = current_memory_usage
        self.max_memory_usage = max_memory_usage

    @classmethod
    def load(cls, job, data, run_path):
        start_time = _deserialize_date(data["start_time"])
        end_time = _deserialize_date(data["end_time"])
        name = data["name"]
        status = data["status"]
        error = data["error"]
        current_memory_usage = data["current_memory_usage"]
        max_memory_usage = data["max_memory_usage"]
        std_err_file = job["stderr"]
        std_out_file = job["stdout"]
        return cls(
            name,
            start_time=start_time,
            end_time=end_time,
            status=status,
            error=error,
            std_out_file=os.path.join(run_path, std_out_file),
            std_err_file=os.path.join(run_path, std_err_file),
            current_memory_usage=current_memory_usage,
            max_memory_usage=max_memory_usage,
        )

    def __str__(self):
        return "name:{} start_time:{}  end_time:{}  status:{}  error:{} ".format(
            self.name, self.start_time, self.end_time, self.status, self.error
        )

    def dump_data(self):
        return {
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "start_time": _serialize_date(self.start_time),
            "end_time": _serialize_date(self.end_time),
            "stdout": self.std_out_file,
            "stderr": self.std_err_file,
            "current_memory_usage": self.current_memory_usage,
            "max_memory_usage": self.max_memory_usage,
        }


class ForwardModelStatus(object):
    def __init__(self, run_id, start_time, end_time=None):
        self.run_id = run_id
        self.start_time = start_time
        self.end_time = end_time
        self._jobs = []

    @classmethod
    def try_load(cls, path):
        status_file = os.path.join(path, File.STATUS_json)
        jobs_file = os.path.join(path, JOBS_FILE)

        with open(status_file) as status_fp:
            status_data = json.load(status_fp)

        with open(jobs_file) as jobs_fp:
            job_data = json.load(jobs_fp)

        start_time = _deserialize_date(status_data["start_time"])
        end_time = _deserialize_date(status_data["end_time"])
        status = cls(status_data["run_id"], start_time, end_time=end_time)

        for job, state in zip(job_data["jobList"], status_data["jobs"]):
            status.add_job(ForwardModelJobStatus.load(job, state, path))

        return status

    @classmethod
    def load(cls, path, num_retry=10):
        sleep_time = 0.10
        attempt = 0

        while attempt < num_retry:
            try:
                status = cls.try_load(path)
                return status
            except (EnvironmentError, ValueError):
                attempt += 1
                if attempt < num_retry:
                    time.sleep(sleep_time)

        return None

    @property
    def jobs(self):
        return self._jobs

    def add_job(self, job):
        self._jobs.append(job)
