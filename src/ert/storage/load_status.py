from enum import Enum
from typing import NamedTuple


class LoadStatus(Enum):
    SUCCESS = 0
    FAILURE = 2


class LoadResult(NamedTuple):
    status: LoadStatus
    message: str
