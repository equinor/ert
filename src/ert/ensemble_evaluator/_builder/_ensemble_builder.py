from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

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
    _reals: List[Realization] = field(default_factory=list)
    _size: int = 0
    _custom_port_range: Optional[range] = None
    _max_running = 10000
    _id: Optional[str] = None

    def add_realization(self, real: Realization) -> Self:
        self._reals.append(real)
        return self

    def set_ensemble_size(self, size: int) -> Self:
        """Duplicate the ensemble members that existed at build time so as to
        get the desired state."""
        self._size = size
        return self

    def set_custom_port_range(self, custom_port_range: range) -> Self:
        self._custom_port_range = custom_port_range
        return self

    def set_max_running(self, max_running: int) -> Self:
        self._max_running = max_running
        return self

    def set_id(self, id_: str) -> Self:
        self._id = id_
        return self

    def build(self) -> Ensemble:
        if not self._reals:
            raise ValueError("Realizations must be added upfront")

        if self._id is None:
            raise ValueError("ID must be set prior to building")

        # legacy has dummy IO, so no need to build an IO map
        if not self.queue_config:
            raise ValueError("missing legacy dependencies")

        return LegacyEnsemble(
            self._reals,
            {},
            self.queue_config,
            self.num_required_realizations,
            id_=self._id,
        )
