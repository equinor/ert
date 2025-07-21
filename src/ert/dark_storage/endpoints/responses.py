import io
import json
from collections.abc import Callable
from typing import Annotated, Any
from urllib.parse import unquote
from uuid import UUID

import pandas as pd
import polars as pl
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, status
from fastapi.responses import Response

from ert.dark_storage.common import get_storage
from ert.storage import Ensemble, Storage

router = APIRouter(tags=["responses"])

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
    filter_on: str | None = Query(None, description="JSON string with filters"),
    accept: Annotated[str | None, Header()] = None,
) -> Response:
    ensemble = storage.get_ensemble(ensemble_id)
    try:
        unquoted_rkey = unquote(response_key)
        dataframe = data_for_response(
            ensemble,
            unquoted_rkey,
            json.loads(filter_on) if filter_on is not None else None,
        )
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
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


# indexing below is based on observation ds columns:
# [ "observation_key", "response_key", *primary_key ]
# for gen_data primary_key is ["report_step", "index"]
# for summary it is ["time"]
response_to_pandas_x_axis_fns: dict[str, Callable[[tuple[Any, ...]], Any]] = {
    "summary": lambda t: pd.Timestamp(t[2]).isoformat(),
    "gen_data": lambda t: str(t[3]),
}


def _extract_response_type_and_key(
    key: str, response_key_to_response_type: dict[str, str]
) -> tuple[str, str] | tuple[None, None]:
    # Check for exact match first. For example if key is "FOPRH"
    # it may stop at "FOPR", which would be incorrect
    response_key = next((k for k in response_key_to_response_type if k == key), None)
    if response_key is None:
        response_key = next(
            (k for k in response_key_to_response_type if k in key and key != f"{k}H"),
            None,
        )

    if response_key is None:
        return None, None

    response_type = response_key_to_response_type.get(response_key)
    assert response_type is not None

    return response_key, response_type


def data_for_response(
    ensemble: Ensemble, key: str, filter_on: dict[str, Any] | None = None
) -> pd.DataFrame:
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
        data = df.unstack(level="Date")
        data.columns = data.columns.droplevel(0)
        return data.astype(float)

    if response_type == "gen_data":
        data = ensemble.load_responses(response_key, tuple(realizations_with_responses))

        try:
            assert filter_on is not None
            assert "report_step" in filter_on
            report_step = int(filter_on["report_step"])
            vals = data.filter(pl.col("report_step").eq(report_step))
            pivoted = vals.drop("response_key", "report_step").pivot(
                on="index", values="values"
            )
            data = pivoted.to_pandas().set_index("realization")
            data.columns = data.columns.astype(int)
            data.columns.name = "axis"
            return data.astype(float)

        except (ValueError, KeyError, pl.ColumnNotFoundError):
            return pd.DataFrame()
