from typing import List

try:
    from typing import TypedDict  # >=3.8
except ImportError:
    from mypy_extensions import TypedDict  # <=3.7


class Argument(TypedDict):
    active_realizations: List[int]
    target_case: str
    analysis_module: str
    weights: str
    start_iteration: int
    prev_successful_realizations: int
    iter_num: int
