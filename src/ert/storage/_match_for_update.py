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


def _join_observations_and_responses(
    strategy: MatchStrategyType,
    observations: pl.DataFrame,
    responses: pl.DataFrame,
    real: int,
) -> pl.DataFrame:
    if responses.is_empty():
        # There are no responses for this realization,
        # so we explicitly create a column of Nones
        # to represent this. We are basically saying that
        # for this realization, each observation points
        # to a None response.
        joined = observations.with_columns(
            pl.lit(None, dtype=pl.Float32).alias(str(real)),
        )
    else:
        joined = strategy.match(observations, responses)

    joined = ensure_qc_error_column(joined)
    joined = _update_qc_error_on_mismatch(joined, strategy.response_cls, str(real))
    joined = _update_realization_column_on_error(joined, str(real))
    joined = _add_index_column(joined, strategy.response_cls, str(real))
    return joined  # ruff: ignore[unnecessary-assign]


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


def _add_index_column(
    joined: pl.DataFrame, response_cls: ResponseConfig, real: str
) -> pl.DataFrame:
    # Avoid potential collision with "index" column (it could be a part
    # of the index_key)
    joined = joined.with_columns(
        response_cls.index_column_expr().alias("__tmp_index_key__")
    )
    if "index" in joined.columns:
        joined = joined.drop("index")
    return joined.rename({"__tmp_index_key__": "index"})


def _update_realization_column_on_error(
    joined: pl.DataFrame, real: str
) -> pl.DataFrame:
    return joined.with_columns(
        pl.when(pl.col(f"qc_error_{real}").is_null())
        .then(pl.col(str(real)))
        .otherwise(pl.lit(None, dtype=pl.Float32))
        .alias(str(real))
    )
