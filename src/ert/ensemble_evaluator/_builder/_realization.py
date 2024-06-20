from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from ert.config.forward_model_step import ForwardModelStep

if TYPE_CHECKING:
    from ert.run_arg import RunArg


@dataclass
class Realization:
    iens: int
    forward_models: Sequence[ForwardModelStep]
    active: bool
    max_runtime: Optional[int]
    run_arg: "RunArg"
    num_cpu: int
    job_script: str
