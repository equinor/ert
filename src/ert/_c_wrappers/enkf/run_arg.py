from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ert.job_queue import RunStatusType
    from ert.storage import EnsembleAccessor


@dataclass
class RunArg:
    run_id: str
    ensemble_storage: EnsembleAccessor
    iens: int
    itr: int
    runpath: str
    job_name: str
    eclbase: str
    active: bool = True
    # Below here is legacy related to Everest
    queue_index: Optional[int] = None
    submitted: bool = False
    run_status: Optional["RunStatusType"] = None

    def set_queue_index(self, index: int) -> None:
        self.queue_index = index

    def getQueueIndex(self) -> int:
        if self.queue_index is None:
            raise ValueError("Queue index not set")
        return self.queue_index

    def isSubmitted(self) -> bool:
        return self.queue_index is not None

    def get_run_id(self) -> str:
        return self.run_id

    @property
    def iter_id(self) -> int:
        return self.itr
