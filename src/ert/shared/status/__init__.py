from typing import NamedTuple, Optional

from ert._c_wrappers.enkf.model_callbacks import LoadStatus


class LoadResult(NamedTuple):
    status: Optional[LoadStatus]
    message: str
