from typing import List, TypedDict


class Argument(TypedDict):
    active_realizations: List[int]
    target_case: str
    analysis_module: str
    weights: str
    start_iteration: int
    prev_successful_realizations: int
    iter_num: int
