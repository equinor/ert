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


class ForwardModelStepStatus:
    def __init__(
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
        cls, fm_step: Dict[str, Any], data: Dict[str, Any], run_path: str
    ) -> "ForwardModelStepStatus":
        start_time = _deserialize_date(data["start_time"])
        end_time = _deserialize_date(data["end_time"])
        name = data["name"]
        status = data["status"]
        error = data["error"]
        current_memory_usage = data["current_memory_usage"]
        max_memory_usage = data["max_memory_usage"]
        std_err_file = fm_step["stderr"]
        std_out_file = fm_step["stdout"]
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
        self._fm_steps: List[ForwardModelStepStatus] = []

    @classmethod
    def try_load(cls, path: str) -> "ForwardModelStatus":
        status_file = os.path.join(path, STATUS_json)
        fm_steps_file = os.path.join(path, JOBS_FILE)

        with open(status_file) as status_fp:
            status_data = json.load(status_fp)

        with open(fm_steps_file) as fm_steps_fp:
            fm_steps_data = json.load(fm_steps_fp)

        start_time = _deserialize_date(status_data["start_time"])
        end_time = _deserialize_date(status_data["end_time"])
        status = cls(status_data["run_id"], start_time, end_time=end_time)

        for fm_step, state in zip(fm_steps_data["jobList"], status_data["jobs"]):
            status.add_step(ForwardModelStepStatus.load(fm_step, state, path))

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
    def steps(self) -> List[ForwardModelStepStatus]:
        return self._fm_steps

    def add_step(self, step: ForwardModelStepStatus) -> None:
        self._fm_steps.append(step)


__all__ = [
    "ForwardModelStepStatus",
    "ForwardModelStatus",
]
