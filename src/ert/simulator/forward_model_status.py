import datetime
import json
import os.path
import time
from typing import Any, Dict, List, Optional

from ert.constant_filenames import JOBS_FILE, STATUS_json


def _serialize_date(dt: Optional[datetime.datetime]) -> Optional[float]:
    if dt is None:
        return None

    return time.mktime(dt.timetuple())


def _deserialize_date(serial_dt: float) -> Optional[datetime.datetime]:
    if serial_dt is None:
        return None

    time_struct = time.localtime(serial_dt)
    return datetime.datetime(*time_struct[0:6])


class ForwardModelJobStatus:  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        status: str = "Waiting",
        error: Optional[str] = None,
        std_out_file: str = "",
        std_err_file: str = "",
        current_memory_usage: int = 0,
        max_memory_usage: int = 0,
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
    def load(
        cls, job: Dict[str, Any], data: Dict[str, Any], run_path: str
    ) -> "ForwardModelJobStatus":
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

    def __str__(self) -> str:
        return (
            f"name:{self.name} start_time:{self.start_time}  "
            f"end_time:{self.end_time}  status:{self.status}  "
            f"error:{self.error} "
        )

    def dump_data(self) -> Dict[str, Any]:
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


class ForwardModelStatus:
    def __init__(
        self,
        run_id: str,
        start_time: Optional[datetime.datetime],
        end_time: Optional[datetime.datetime] = None,
    ):
        self.run_id = run_id
        self.start_time = start_time
        self.end_time = end_time
        self._jobs: List[ForwardModelJobStatus] = []

    @classmethod
    def try_load(cls, path: str) -> "ForwardModelStatus":
        status_file = os.path.join(path, STATUS_json)
        jobs_file = os.path.join(path, JOBS_FILE)

        with open(status_file) as status_fp:  # pylint: disable=unspecified-encoding
            status_data = json.load(status_fp)

        with open(jobs_file) as jobs_fp:  # pylint: disable=unspecified-encoding
            job_data = json.load(jobs_fp)

        start_time = _deserialize_date(status_data["start_time"])
        end_time = _deserialize_date(status_data["end_time"])
        status = cls(status_data["run_id"], start_time, end_time=end_time)

        for job, state in zip(job_data["jobList"], status_data["jobs"]):
            status.add_job(ForwardModelJobStatus.load(job, state, path))

        return status

    @classmethod
    def load(cls, path: str, num_retry: int = 10) -> Optional["ForwardModelStatus"]:
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
    def jobs(self) -> List[ForwardModelJobStatus]:
        return self._jobs

    def add_job(self, job: ForwardModelJobStatus) -> None:
        self._jobs.append(job)


__all__ = [
    "ForwardModelJobStatus",
    "ForwardModelStatus",
]
