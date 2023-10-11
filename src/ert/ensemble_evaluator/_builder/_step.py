from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ert.config.ext_job import ExtJob

if TYPE_CHECKING:
    from ert.run_arg import RunArg


@dataclass
class LegacyJob:
    id_: str
    index: str
    name: str
    ext_job: ExtJob


@dataclass
class LegacyStep:
    name: str
    max_runtime: Optional[int]
    run_arg: "RunArg"
    num_cpu: int
    job_script: str
