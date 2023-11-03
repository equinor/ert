import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from typing_extensions import Self

from ert.config.forward_model import ForwardModel

if TYPE_CHECKING:
    from ert.run_arg import RunArg
SOURCE_TEMPLATE_REAL = "/real/{iens}"

logger = logging.getLogger(__name__)


@dataclass
class Realization:
    iens: int
    forward_models: Sequence[ForwardModel]
    active: bool
    max_runtime: Optional[int]
    run_arg: "RunArg"
    num_cpu: int
    job_script: str


class RealizationBuilder:
    def __init__(self) -> None:
        self._active: Optional[bool] = None
        self._iens: Optional[int] = None
        self._parent_source: Optional[str] = None
        self._forward_models: Sequence[ForwardModel] = []
        self._max_runtime: Optional[int] = None
        self._run_arg: Optional["RunArg"] = None
        self._num_cpu: Optional[int] = None
        self._job_script: Optional[str] = None

    def active(self, active: bool) -> Self:
        self._active = active
        return self

    def set_forward_models(self, forward_models: Sequence[ForwardModel]) -> Self:
        self._forward_models = forward_models
        return self

    def set_iens(self, iens: int) -> Self:
        self._iens = iens
        return self

    def set_max_runtime(self, max_runtime: Optional[int]) -> Self:
        self._max_runtime = max_runtime
        return self

    def set_run_arg(self, run_arg: "RunArg") -> Self:
        self._run_arg = run_arg
        return self

    def set_num_cpu(self, num_cpu: int) -> Self:
        self._num_cpu = num_cpu
        return self

    def set_job_script(self, job_script: str) -> Self:
        self._job_script = job_script
        return self

    def build(self) -> Realization:
        if not self._iens:
            # assume this is being used as a forward model, thus should be 0
            self._iens = 0

        assert self._active is not None

        if self._active:
            assert self._run_arg is not None
            assert self._num_cpu is not None
            assert self._job_script is not None
            # Mypy is not able to pick up these asserts due
            # to the condition on self._active, thus we still
            # need to ignore typing errors below

        return Realization(
            self._iens,
            self._forward_models,
            self._active,
            self._max_runtime,
            self._run_arg,  # type: ignore
            self._num_cpu,  # type: ignore
            self._job_script,  # type: ignore
        )
