"""Utility to compensate for a weak job type."""
import time
from job_runner.reporting.message import (
    Exited,
    Finish,
    Init,
    Running,
    Start,
    _JOB_STATUS_SUCCESS,
    _JOB_STATUS_RUNNING,
    _JOB_STATUS_FAILURE,
    _JOB_STATUS_WAITING,
)


def create_job_dict(job):
    return {
        "name": job.name(),
        "status": _JOB_STATUS_WAITING,
        "error": None,
        "start_time": None,
        "end_time": None,
        "stdout": job.std_out,
        "stderr": job.std_err,
        "current_memory_usage": None,
        "max_memory_usage": None,
    }


def datetime_serialize(dt):
    if dt is None:
        return None
    return time.mktime(dt.timetuple())
