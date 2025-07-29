"""Utility to compensate for a weak step type."""

import time
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from _ert.forward_model_runner.forward_model_step import ForwardModelStep


class StepStatus(StrEnum):
    WAITING = "Waiting"
    RUNNING = "Running"
    FAILURE = "Failure"
    SUCCESS = "Success"


class StepDict(TypedDict, total=False):
    name: str
    status: StepStatus
    error: str | None
    start_time: float | None
    end_time: float | None
    stdout: str | None
    stderr: str | None
    current_memory_usage: int | None
    max_memory_usage: int | None
    cpu_seconds: float


def create_step_dict(step: "ForwardModelStep") -> StepDict:
    return {
        "name": step.name(),
        "status": StepStatus.WAITING,
        "error": None,
        "start_time": None,
        "end_time": None,
        "stdout": step.std_out,
        "stderr": step.std_err,
        "current_memory_usage": None,
        "max_memory_usage": None,
    }


def datetime_serialize(dt: datetime) -> float | None:
    if dt is None:
        return None
    return time.mktime(dt.timetuple())
