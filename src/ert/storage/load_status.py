from enum import Enum
from typing import NamedTuple


class LoadStatus(Enum):
    LOAD_SUCCESSFUL = 0
    LOAD_FAILURE = 2


class LoadResult(NamedTuple):
    status: LoadStatus | None
    message: str
