from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from typing_extensions import Self

from ._legacy import LegacyEnsemble
from ._realization import Realization

if TYPE_CHECKING:
    from ert.config import QueueConfig

    from ._ensemble import Ensemble

logger = logging.getLogger(__name__)


@dataclass
class EnsembleBuilder:
    queue_config: QueueConfig
    num_required_realizations: int
    id: str
    _reals: List[Realization] = field(default_factory=list)

    def add_realization(self, real: Realization) -> Self:
        self._reals.append(real)
        return self

    def build(self) -> Ensemble:
        if not self._reals:
            raise ValueError("Realizations must be added upfront")

        return LegacyEnsemble(
            self._reals,
            {},
            self.queue_config,
            self.num_required_realizations,
            id_=self.id,
        )
