import logging
from typing import List, Optional, Sequence

from typing_extensions import Self

from ._step import LegacyStep

SOURCE_TEMPLATE_REAL = "/real/{iens}"

logger = logging.getLogger(__name__)


class Realization:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        iens: int,
        steps: Sequence[LegacyStep],
        active: bool,
    ):
        if iens is None:
            raise ValueError(f"{self} needs iens")
        if steps is None:
            raise ValueError(f"{self} needs steps")
        if active is None:
            raise ValueError(f"{self} needs to be set either active or not")

        self.iens = iens
        self.steps = steps
        self.active = active


class RealizationBuilder:
    def __init__(self) -> None:
        self._steps: List[LegacyStep] = []
        self._active: Optional[bool] = None
        self._iens: Optional[int] = None
        self._parent_source: Optional[str] = None

    def active(self, active: bool) -> Self:
        self._active = active
        return self

    def add_step(self, step: LegacyStep) -> Self:
        self._steps.append(step)
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
            self._steps,
            self._active,
        )
