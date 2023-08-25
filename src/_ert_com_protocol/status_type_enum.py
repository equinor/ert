from typing import Dict

_queue_state_to_pbuf_type_map: Dict[str, str] = {
    "NOT_ACTIVE": "STEP_WAITING",
    "WAITING": "STEP_WAITING",
    "SUBMITTED": "STEP_WAITING",
    "PENDING": "STEP_PENDING",
    "RUNNING": "STEP_RUNNING",
    "DONE": "STEP_RUNNING",
    "EXIT": "STEP_RUNNING",
    "IS_KILLED": "STEP_FAILED",
    "DO_KILL": "STEP_FAILED",
    "SUCCESS": "STEP_SUCCESS",
    "RUNNING_DONE_CALLBACK": "STEP_RUNNING",
    "RUNNING_EXIT_CALLBACK": "STEP_RUNNING",
    "STATUS_FAILURE": "STEP_UNKNOWN",
    "FAILED": "STEP_FAILED",
    "UNKNOWN": "STEP_UNKNOWN",
}


def queue_state_to_pbuf_type(status: str) -> str:
    return _queue_state_to_pbuf_type_map[status]
