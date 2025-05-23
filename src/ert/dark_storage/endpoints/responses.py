import io
from typing import Annotated
from urllib.parse import unquote
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Header, HTTPException, status
from fastapi.responses import Response

from ert.dark_storage import json_schema as js
from ert.dark_storage.common import data_for_response, get_storage
from ert.storage import Storage

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
def get_response(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
    response_key: str,
    accept: Annotated[str | None, Header()] = None,
) -> list[js.ObservationOut]:
    ensemble = storage.get_ensemble(ensemble_id)
    try:
        unquoted_rkey = unquote(response_key)
        dataframe = data_for_response(ensemble, unquoted_rkey)
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
