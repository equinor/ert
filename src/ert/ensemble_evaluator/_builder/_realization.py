import logging
from typing import Optional, Sequence

from typing_extensions import Self

from ._step import LegacyJob, LegacyStep

SOURCE_TEMPLATE_REAL = "/real/{iens}"

logger = logging.getLogger(__name__)


class Realization:
    def __init__(
        self,
        iens: int,
        step: Optional[LegacyStep],
        jobs: Sequence[LegacyJob],
        active: bool,
    ):
        if iens is None:
            raise ValueError(f"{self} needs iens")
        if jobs is None:
            raise ValueError(f"{self} needs jobs")
        if active is None:
            raise ValueError(f"{self} needs to be set either active or not")

        self.iens = iens
        self.step = step
        self.jobs = jobs
        self.active = active


class RealizationBuilder:
    def __init__(self) -> None:
        self._step: Optional[LegacyStep] = None
        self._active: Optional[bool] = None
        self._iens: Optional[int] = None
        self._parent_source: Optional[str] = None
        self._jobs: Sequence[LegacyJob] = []

    def active(self, active: bool) -> Self:
        self._active = active
        return self

    def set_step(self, step: LegacyStep) -> Self:
        self._step = step
        return self

    def set_jobs(self, jobs: Sequence[LegacyJob]) -> Self:
        self._jobs = jobs
        return self

    def set_iens(self, iens: int) -> Self:
        self._iens = iens
        return self

    def build(self) -> Realization:
        if not self._iens:
            # assume this is being used as a forward model, thus should be 0
            self._iens = 0

        if self._active is None:
            raise ValueError(f"realization {self._iens}: active should be set")

        return Realization(
            self._iens,
            self._step,
            self._jobs,
            self._active,
        )
