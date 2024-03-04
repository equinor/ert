from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ert.storage import Ensemble


@dataclass
class RunArg:
    run_id: str
    ensemble_storage: Ensemble
    iens: int
    itr: int
    runpath: str
    job_name: str
    active: bool = True
    # Below here is legacy related to Everest
    queue_index: Optional[int] = None
    submitted: bool = False
