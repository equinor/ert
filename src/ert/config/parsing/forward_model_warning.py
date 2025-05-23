from typing import TypedDict


class ForwardModelWarning(TypedDict):
    warning_msg: str
    warning_count: int
    iens: int
    step_name: str
    step_idx: int
    filetype: str
