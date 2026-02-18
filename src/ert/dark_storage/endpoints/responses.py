import io
import json
import logging
from collections.abc import Callable
from typing import Annotated, Any
from urllib.parse import unquote
from uuid import UUID

import pandas as pd
import polars as pl
from fastapi import APIRouter, Body, Depends, Header, Query, status
from fastapi.responses import Response
from polars.exceptions import ColumnNotFoundError

from ert.dark_storage.common import get_storage, reraise_as_http_errors
from ert.storage import Ensemble, Storage

router = APIRouter(tags=["responses"])
logger = logging.getLogger(__name__)


DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)


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

    media_type = accept if accept is not None else "text/csv"
    if media_type == "application/x-parquet":
        dataframe.columns = [str(s) for s in dataframe.columns]
        stream = io.BytesIO()
        dataframe.to_parquet(stream)
        return Response(
            content=stream.getvalue(),
            media_type="application/x-parquet",
        )
    elif media_type == "application/json":
        return Response(dataframe.to_json(), media_type="application/json")
    else:
        return Response(
            content=dataframe.to_csv().encode(),
            media_type="text/csv",
        )


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

    media_type = accept if accept is not None else "text/csv"
    if media_type == "application/x-parquet":
        dataframe.columns = [str(s) for s in dataframe.columns]
        stream = io.BytesIO()
        dataframe.to_parquet(stream)
        return Response(
            content=stream.getvalue(),
            media_type="application/x-parquet",
        )
    elif media_type == "application/json":
        return Response(dataframe.to_json(), media_type="application/json")
    else:
        return Response(
            content=dataframe.to_csv().encode(),
            media_type="text/csv",
        )


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
# [ "observation_key", "response_key", *primary_key ]
# for gen_data primary_key is ["report_step", "index"]
# for summary it is ["time"]
response_to_pandas_x_axis_fns: dict[str, Callable[[tuple[Any, ...]], Any]] = {
    "summary": lambda t: pd.Timestamp(t[2]).isoformat(),
    "gen_data": lambda t: str(t[3]),
    "rft": lambda t: str(t[4]),
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

        df = ensemble.batch_objectives.clone()
        improvements = {}
        for ens in ensemble.experiment.ensembles:
            improvements[ens.iteration] = ens.is_improvement

        imp_df = pl.DataFrame(
            {
                "batch_id": list(improvements.keys()),
                "is_improvement": list(improvements.values()),
            },
            schema={"batch_id": pl.Int64, "is_improvement": pl.Boolean},
        )

        df = df.join(imp_df, on="batch_id", how="left")
        df = df.with_columns(pl.col("is_improvement").fill_null(False))

        return (
            df.select(["batch_id", "total_objective_value", "is_improvement"])
            .to_pandas()
            .astype(
                {
                    "batch_id": int,
                    "total_objective_value": float,
                    "is_improvement": bool,
                }
            )
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

    if response_type == "summary":
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
        # This performs the same aggragation by mean of duplicate values
        # as in ert/analysis/_es_update.py
        df = df.groupby(["Date", "Realization"]).mean()
        data = df.reset_index().pivot_table(
            index="Realization", columns="Date", values=df.columns[0]
        )
        return data.astype(float)

    if response_type == "rft":
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
            .reset_index(drop=True)
        )

    if response_type == "gen_data":
        data = ensemble.load_responses(response_key, tuple(realizations_with_responses))

        try:
            assert filter_on is not None
            assert "report_step" in filter_on
            report_step = int(filter_on["report_step"])
            vals = data.filter(pl.col("report_step").eq(report_step))
            pivoted = vals.drop("response_key", "report_step").pivot(  # noqa: PD010
                on="index", values="values"
            )
            data = pivoted.to_pandas().set_index("realization")
            data.columns = data.columns.astype(int)
            data.columns.name = "axis"
            return data.astype(float)

        except (ValueError, KeyError, ColumnNotFoundError):
            return pd.DataFrame()

    if response_type in {"everest_objectives", "everest_constraints"}:
        df_pl = (
            ensemble.realization_objectives
            if response_type == "everest_objectives"
            else ensemble.realization_constraints
        )
        if df_pl is None or response_key not in df_pl.columns:
            return pd.DataFrame()

        columns = ["batch_id", "realization", response_key]
        return df_pl.select(columns).to_pandas()

    return pd.DataFrame()
