"""Utility to compensate for a weak step type."""

import time

from _ert.forward_model_runner.reporting.message import _STEP_STATUS_WAITING


def create_step_dict(step):
    return {
        "name": step.name(),
        "status": _STEP_STATUS_WAITING,
        "error": None,
        "start_time": None,
        "end_time": None,
        "stdout": step.std_out,
        "stderr": step.std_err,
        "current_memory_usage": None,
        "max_memory_usage": None,
    }


def datetime_serialize(dt):
    if dt is None:
        return None
    return time.mktime(dt.timetuple())
