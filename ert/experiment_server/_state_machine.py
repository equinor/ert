from collections import defaultdict
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ExperimentStateMachine:
    """The :class:`ExperimentStateMachine` implements a state machine for the
    entire experiment. It allows an experiment to track its own state and
    communicate it to others."""

    def __init__(self) -> None:
        self._ensemble_to_successful_realizations: Dict[int, List[int]] = defaultdict(
            list
        )

    def successful_realizations(self, iter_: int) -> int:
        """Return an integer indicating the number of successful realizations
        in an ensemble given ``iter_``. Raise :class:`IndexError` if the
        ensemble has no successful realizations."""
        return len(self._ensemble_to_successful_realizations[iter_])

    def add_successful_realization(self, iter_: int, real: int) -> None:
        """Add a successful realization for realization ``real`` for iteration
        ``iter_``."""
        logger.debug("adding successful real for iter %d, real: %d", iter_, real)
        if real not in self._ensemble_to_successful_realizations[iter_]:
            self._ensemble_to_successful_realizations[iter_].append(real)
