from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, List, Optional

from typing_extensions import Self

from ._legacy import LegacyEnsemble
from ._realization import RealizationBuilder

if TYPE_CHECKING:
    from ert.config import QueueConfig

    from ._ensemble import Ensemble

logger = logging.getLogger(__name__)


class EnsembleBuilder:
    def __init__(self) -> None:
        self._reals: List[RealizationBuilder] = []
        self._forward_model: Optional[RealizationBuilder] = None
        self._size: int = 0
        self._legacy_dependencies: Optional["QueueConfig"] = None
        self.stop_long_running = False
        self.num_required_realizations = 0
        self._custom_port_range: Optional[range] = None
        self._max_running = 10000
        self._id: Optional[str] = None

    def set_forward_model(self, forward_model: RealizationBuilder) -> Self:
        if self._reals:
            raise ValueError(
                "Cannot set forward model when realizations are already specified"
            )
        self._forward_model = forward_model
        return self

    def add_realization(self, real: RealizationBuilder) -> Self:
        if self._forward_model:
            raise ValueError("Cannot add realization when forward model is specified")

        self._reals.append(real)
        return self

    def set_ensemble_size(self, size: int) -> Self:
        """Duplicate the ensemble members that existed at build time so as to
        get the desired state."""
        self._size = size
        return self

    def set_legacy_dependencies(
        self,
        queue_config: QueueConfig,
        stop_long_running: bool,
        num_required_realizations: int,
    ) -> Self:
        self._legacy_dependencies = queue_config
        self.stop_long_running = stop_long_running
        self.num_required_realizations = num_required_realizations
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
        if not (self._reals or self._forward_model):
            raise ValueError("Either forward model or realizations needs to be set")

        if self._id is None:
            raise ValueError("ID must be set prior to building")

        real_builders: List[RealizationBuilder] = []
        if self._forward_model:
            # duplicate the original forward model into realizations
            for i in range(self._size):
                logger.debug(f"made deep-copied real {i}")
                real = copy.deepcopy(self._forward_model)
                real.set_iens(i)
                real_builders.append(real)
        else:
            real_builders = self._reals

        # legacy has dummy IO, so no need to build an IO map
        if not self._legacy_dependencies:
            raise ValueError("missing legacy dependencies")

        reals = [builder.build() for builder in real_builders]

        return LegacyEnsemble(
            reals,
            {},
            self._legacy_dependencies,
            self.stop_long_running,
            self.num_required_realizations,
            id_=self._id,
        )
