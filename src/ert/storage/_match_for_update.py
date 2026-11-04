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


def _update_qc_error_on_mismatch(
    joined: pl.DataFrame, response_cls: ResponseConfig, real: str
) -> pl.DataFrame:
    no_matched_response_condition = pl.col(str(real)).is_null()
    no_matched_response_error = pl.concat_str(
        [
            pl.lit("no response matched observation data: "),
            pl.lit("response_key="),
            pl.col("response_key").cast(pl.String),
            pl.lit(", "),
            response_cls.match_key_dict_expr(),
        ],
        separator="",
    )

    return joined.with_columns(
        append_to_qc_error(
            no_matched_response_condition, no_matched_response_error
        ).alias(f"qc_error_{real}")
    )
