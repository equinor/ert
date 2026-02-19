import io
import logging
from typing import Annotated
from urllib.parse import unquote
from uuid import UUID

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Depends, Header, HTTPException, status
from fastapi.responses import Response

from ert.dark_storage.common import get_storage, reraise_as_http_errors
from ert.storage import Ensemble, Storage

router = APIRouter(tags=["ensemble"])

DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)

router = APIRouter(tags=["parameters"])
logger = logging.getLogger(__name__)


@router.get(
    "/ensembles/{ensemble_id}/parameters/{parameter_key}",
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
async def get_parameter(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
    parameter_key: str,
    accept: Annotated[str | None, Header()] = None,
) -> Response:
    with reraise_as_http_errors(logger):
        ensemble = storage.get_ensemble(ensemble_id)
        unquoted_pkey = unquote(parameter_key)
        dataframe = data_for_parameter(ensemble, unquoted_pkey)

    match accept:
        case "application/x-parquet":
            dataframe.columns = [str(s) for s in dataframe.columns]
            stream = io.BytesIO()
            dataframe.to_parquet(stream)
            return Response(
                content=stream.getvalue(),
                media_type="application/x-parquet",
            )
        case "application/json":
            return Response(dataframe.to_json(), media_type="application/json")
        case "text/csv" | None:
            return Response(
                content=dataframe.to_csv().encode(),
                media_type="text/csv",
            )
    raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)


@router.get("/ensembles/{ensemble_id}/parameters/{key}/std_dev")
def get_parameter_std_dev(
    *, storage: Storage = DEFAULT_STORAGE, ensemble_id: UUID, key: str, z: int
) -> Response:
    key = unquote(key)
    with reraise_as_http_errors(logger):
        ensemble = storage.get_ensemble(ensemble_id)
        da = ensemble.calculate_std_dev_for_parameter_group(key)

    if z >= int(da.shape[2]):
        logger.error("invalid z index")
        raise HTTPException(status_code=500, detail="Internal server error")

    data_2d = da[:, :, z]

    buffer = io.BytesIO()
    np.save(buffer, data_2d)

    return Response(content=buffer.getvalue(), media_type="application/octet-stream")


def data_for_parameter(ensemble: Ensemble, key: str) -> pd.DataFrame:
    param_info = ensemble.experiment.parameter_info.get(key)

    if (
        param_info
        and param_info.get("type") == "everest_parameters"
        and (everest_controls := ensemble.realization_controls) is not None
        and key in everest_controls.columns
    ):
        columns_to_select = [
            col
            for col in ["batch_id", "realization", key]
            if col in everest_controls.columns
        ]

        return everest_controls.select(columns_to_select).to_pandas()

    with reraise_as_http_errors(logger):
        try:
            df = ensemble.load_scalar_keys([key], transformed=True)
            if df.is_empty():
                logger.warning(f"No data found for parameter '{key}'")
                return pd.DataFrame()
        except KeyError as e:
            logger.error(e)
            return pd.DataFrame()

    dataframe = df.to_pandas().set_index("realization")
    dataframe.columns.name = None
    dataframe.index.name = "Realization"
    data = dataframe.sort_index(axis=1)
    if data.empty or key not in data:
        return pd.DataFrame()
    data = data[key].to_frame().dropna()
    data.columns = pd.Index([0])
    return data
