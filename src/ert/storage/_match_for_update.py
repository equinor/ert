from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ert.config.observation_quality_control import (
    append_to_qc_error,
    ensure_qc_error_column,
)
from ert.config.response_config import ResponseConfig

if TYPE_CHECKING:
    import numpy.typing as npt

    from .local_ensemble import LocalEnsemble
from ert.storage.match import MatchStrategyType, matching_strategy


def _validate_selected_observations(
    ensemble: LocalEnsemble, selected_observations: Iterable[str]
) -> None:
    known_observations = ensemble.experiment.observation_keys
    unknown_observations = set(selected_observations) - set(known_observations)
    if unknown_observations:
        raise KeyError(
            f"Observations: {', '.join(unknown_observations)} not in experiment"
        )
