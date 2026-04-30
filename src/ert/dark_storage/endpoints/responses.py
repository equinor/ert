import json
import logging
from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Any
from urllib.parse import unquote
from uuid import UUID

import numpy as np
import pandas as pd
import polars as pl
from fastapi import APIRouter, Body, Depends, Header, Query, status
from fastapi.responses import Response
from polars.exceptions import ColumnNotFoundError

from ert.dark_storage.common import (
    get_storage,
    reraise_as_http_errors,
    serialize_dataframe_to_response,
)
from ert.storage import Ensemble, Storage

router = APIRouter(tags=["responses"])
logger = logging.getLogger(__name__)


DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)


class RejectionReason(StrEnum):
    NON_IMPROVEMENT = "non-improvement"
    BOUND_CONSTRAINT_VIOLATION = "bound constraint violation"
    INPUT_CONSTRAINT_VIOLATION = "input constraint violation"
    OUTPUT_CONSTRAINT_VIOLATION = "output constraint violation"


@router.get(
    "/ensembles/{ensemble_id}/responses/{response_key}",
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {},
                "text/csv": {},
                "application/x-parquet": {},
            }
        },
        status.HTTP_401_UNAUTHORIZED: {
            "content": {
                "application/json": {"example": {"error": "Unauthorized access"}}
            },
        },
    },
)
async def get_response(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
    response_key: str,
    filter_on: Annotated[
        str | None, Query(description="JSON string with filters")
    ] = None,
    accept: Annotated[str | None, Header()] = None,
) -> Response:
    with reraise_as_http_errors(logger):
        ensemble = storage.get_ensemble(ensemble_id)
        unquoted_rkey = unquote(response_key)
        dataframe = data_for_response(
            ensemble,
            unquoted_rkey,
            json.loads(filter_on) if filter_on is not None else None,
        )
    return serialize_dataframe_to_response(dataframe, accept)


@router.get(
    "/ensembles/{ensemble_id}/gradients/{key}",
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {},
                "text/csv": {},
                "application/x-parquet": {},
            }
        },
        status.HTTP_401_UNAUTHORIZED: {
            "content": {
                "application/json": {"example": {"error": "Unauthorized access"}}
            },
        },
    },
)
async def get_gradient(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
    key: str,
    accept: Annotated[str | None, Header()] = None,
) -> Response:
    with reraise_as_http_errors(logger):
        ensemble = storage.get_ensemble(ensemble_id)
        unquoted_key = unquote(key)
        dataframe = data_for_gradient(ensemble, unquoted_key)

    return serialize_dataframe_to_response(dataframe, accept)


def data_for_gradient(ensemble: Ensemble, key: str) -> pd.DataFrame:
    response_key, response_type = _extract_response_type_and_key(
        key, ensemble.experiment.response_key_to_response_type
    )

    if response_key is None:
        return pd.DataFrame()

    df = None
    if response_type == "everest_objectives":
        df = ensemble.batch_objective_gradient
    elif response_type == "everest_constraints":
        df = ensemble.batch_constraint_gradient

    if df is None:
        return pd.DataFrame()

    if response_key not in df.columns:
        return pd.DataFrame()

    return (
        df.select(["batch_id", "control_name", response_key])
        .to_pandas()
        .astype(
            {
                "batch_id": int,
                "control_name": str,
                response_key: float,
            }
        )
    )


# indexing below is based on observation ds columns:
# [ "observation_key", "response_key", *match_key ]
# for gen_data match_key is ["report_step", "index"]
# for summary it is ["time"]
response_to_pandas_x_axis_fns: dict[str, Callable[[tuple[Any, ...]], Any]] = {
    "summary": lambda t: pd.Timestamp(t[2]).isoformat(),
    "gen_data": lambda t: str(t[3]),
    "rft": lambda t: str(t[6]),
    "breakthrough": lambda t: pd.Timestamp(t[2]).isoformat(),
}


def _extract_response_type_and_key(
    key: str, response_key_to_response_type: dict[str, str]
) -> tuple[str, str] | tuple[None, None]:
    # Only allow exact matches.
    response_key = key if key in response_key_to_response_type else None

    if response_key is None:
        return None, None

    response_type = response_key_to_response_type.get(response_key)
    assert response_type is not None

    return response_key, response_type


def data_for_response(
    ensemble: Ensemble, key: str, filter_on: dict[str, Any] | None = None
) -> pd.DataFrame | pd.Series:
    if key == "total objective value":
        if ensemble.batch_objectives is None:
            return pd.DataFrame()

        def constraint_violation_check(violation: pl.DataFrame | None) -> float:
            if violation is None:
                return 0.0
            return violation.drop("batch_id").to_numpy().max().item()

        CONSTRAINT_TOL = 1e-6
        accepted_batches: list[tuple[Ensemble, float]] = []
        rejected_batches_with_value_and_reason: list[
            tuple[Ensemble, float, RejectionReason]
        ] = []
        max_total_objective = np.inf
        for current_ensemble in ensemble.experiment.ensembles_with_function_results:
            if current_ensemble.batch_objectives is None:
                raise ValueError(
                    f"Ensemble {current_ensemble.id} does not have batch objectives"
                )
            total_objective = current_ensemble.batch_objectives[
                "total_objective_value"
            ].item()

            bound_violation = constraint_violation_check(
                current_ensemble.batch_bound_constraint_violations
            )
            input_violation = constraint_violation_check(
                current_ensemble.batch_input_constraint_violations
            )
            output_violation = constraint_violation_check(
                current_ensemble.batch_output_constraint_violations
            )

            if (
                max(
                    bound_violation,
                    input_violation,
                    output_violation,
                )
                < CONSTRAINT_TOL
                and total_objective < max_total_objective
            ):
                accepted_batches.append(
                    (
                        current_ensemble,
                        (max_total_objective - total_objective)
                        if max_total_objective != np.inf
                        else 0.0,
                    )
                )
                max_total_objective = total_objective
            elif total_objective >= max_total_objective:
                rejected_batches_with_value_and_reason.append(
                    (
                        current_ensemble,
                        total_objective - max_total_objective,
                        RejectionReason.NON_IMPROVEMENT,
                    )
                )

            else:
                violations = {
                    RejectionReason.BOUND_CONSTRAINT_VIOLATION: bound_violation,
                    RejectionReason.INPUT_CONSTRAINT_VIOLATION: input_violation,
                    RejectionReason.OUTPUT_CONSTRAINT_VIOLATION: output_violation,
                }
                rejection_reason = max(violations, key=lambda k: violations[k])
                rejected_batches_with_value_and_reason.append(
                    (
                        current_ensemble,
                        violations[rejection_reason],
                        rejection_reason,
                    )
                )

        objective_value_df = ensemble.batch_objectives.clone().select(
            ["batch_id", "total_objective_value"]
        )

        improvement_df = objective_value_df.join(
            pl.DataFrame(
                {
                    "batch_id": [batch.iteration for batch, _ in accepted_batches],
                    "is_improvement": [True] * len(accepted_batches),
                    "improvement_value": [value for _, value in accepted_batches],
                }
            ),
            on="batch_id",
            how="left",
        ).with_columns(
            pl.col("total_objective_value").neg(),
            pl.col("is_improvement").fill_null(False),
        )

        if rejected_batches_with_value_and_reason:
            batches, values, reasons = zip(
                *rejected_batches_with_value_and_reason, strict=True
            )
            rejected_df = pl.DataFrame(
                {
                    "batch_id": [b.iteration for b in batches],
                    "constraint_violation_value": list(values),
                    "constraint_violation_type": [r.value for r in reasons],
                }
            )
            return (
                improvement_df.join(rejected_df, on="batch_id", how="left")
                .to_pandas()
                .astype(
                    {
                        "batch_id": int,
                        "total_objective_value": float,
                        "is_improvement": bool,
                        "improvement_value": float,
                        "constraint_violation_value": float,
                        "constraint_violation_type": str,
                    }
                )
            )
        return improvement_df.to_pandas().astype(
            {
                "batch_id": int,
                "total_objective_value": float,
                "is_improvement": bool,
                "improvement_value": float,
            }
        )

    response_key, response_type = _extract_response_type_and_key(
        key, ensemble.experiment.response_key_to_response_type
    )

    if response_key is None:
        return pd.DataFrame()

    assert response_key is not None
    assert response_type is not None

    realizations_with_responses = ensemble.get_realization_list_with_responses()

    if len(realizations_with_responses) == 0:
        return pd.DataFrame()

    match response_type:
        case "summary" | "breakthrough":
            summary_data = ensemble.load_responses(
                response_key,
                tuple(realizations_with_responses),
            )

            df = (
                summary_data.rename({"time": "Date", "realization": "Realization"})
                .drop("response_key")
                .to_pandas()
            )
            df = df.set_index(["Date", "Realization"])
            # This performs the same aggregation by mean of duplicate values
            # as in ert/analysis/_es_update.py
            df = df.groupby(["Date", "Realization"]).mean()
            summary_value_col = 0
            breakthrough_value_col = 1
            value_column = (
                summary_value_col
                if response_type == "summary"
                else breakthrough_value_col
            )
            data = df.reset_index().pivot_table(
                index="Realization", columns="Date", values=df.columns[value_column]
            )
            return data.astype(float)
        case "rft":
            return (
                ensemble.load_responses(
                    response_key,
                    tuple(realizations_with_responses),
                )
                .rename({"realization": "Realization"})
                .select(["Realization", "depth", "values"])
                .unique()
                .to_pandas()
                .pivot_table(index="Realization", columns="depth", values="values")
            )
        case "gen_data":
            data = ensemble.load_responses(
                response_key, tuple(realizations_with_responses)
            )

            try:
                assert filter_on is not None
                assert "report_step" in filter_on
                report_step = int(filter_on["report_step"])
                vals = data.filter(pl.col("report_step").eq(report_step))
                pivoted = vals.drop("response_key", "report_step").pivot(  # noqa: PD010
                    on="index", values="values"
                )
                data = (
                    pivoted.rename({"realization": "Realization"})
                    .to_pandas()
                    .set_index("Realization")
                )
                data.columns = data.columns.astype(int)
                data.columns.name = "axis"
                return data.astype(float)

            except (ValueError, KeyError, ColumnNotFoundError):
                return pd.DataFrame()
        case "everest_objectives" | "everest_constraints":
            df_pl = (
                ensemble.realization_objectives
                if response_type == "everest_objectives"
                else ensemble.realization_constraints
            )
            if df_pl is None or response_key not in df_pl.columns:
                return pd.DataFrame()

            columns = ["batch_id", "realization", response_key]

            if response_type == "everest_constraints":
                constraints = ensemble.experiment.output_constraints
                if constraints is not None and response_key in constraints.keys:
                    idx = constraints.keys.index(response_key)
                    return (
                        df_pl.select(columns)
                        .with_columns(
                            pl.lit(constraints.lower_bounds[idx]).alias("lower_bound"),
                            pl.lit(constraints.upper_bounds[idx]).alias("upper_bound"),
                        )
                        .to_pandas()
                    )
            return df_pl.select(columns).to_pandas()
        case _:
            return pd.DataFrame()
