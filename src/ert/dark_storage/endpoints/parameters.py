import io
import logging
from typing import Annotated
from urllib.parse import unquote
from uuid import UUID

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Depends, Header, HTTPException, status
from fastapi.responses import Response

from ert.dark_storage.common import get_storage
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
    try:
        ensemble = storage.get_ensemble(ensemble_id)
    except KeyError as e:
        logger.error(e)
        raise HTTPException(status_code=404, detail="Ensemble not found") from e
    except Exception as ex:
        logger.exception(ex)
        raise HTTPException(status_code=500, detail="Internal server error") from ex

    unquoted_pkey = unquote(parameter_key)

    try:
        dataframe = data_for_parameter(ensemble, unquoted_pkey)
    except PermissionError as e:
        logger.error(e)
        raise HTTPException(status_code=401, detail=str(e)) from e
    except Exception as ex:
        logger.exception(ex)
        raise HTTPException(status_code=500, detail="Internal server error") from ex

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


@router.get("/ensembles/{ensemble_id}/parameters/{key}/std_dev")
def get_parameter_std_dev(
    *, storage: Storage = DEFAULT_STORAGE, ensemble_id: UUID, key: str, z: int
) -> Response:
    key = unquote(key)
    try:
        ensemble = storage.get_ensemble(ensemble_id)
        da = ensemble.calculate_std_dev_for_parameter_group(key)
    except ValueError as e:
        logger.error(e)
        raise HTTPException(status_code=404, detail="Data not found") from e
    except KeyError as ke:
        logger.error(ke)
        raise HTTPException(status_code=404, detail="Ensemble not found") from ke
    except Exception as ex:
        logger.exception(ex)
        raise HTTPException(status_code=500, detail="Internal server error") from ex

    if z >= int(da.shape[2]):
        logger.error("invalid z index")
        raise HTTPException(status_code=500, detail="Internal server error")

    data_2d = da[:, :, z]

    buffer = io.BytesIO()
    np.save(buffer, data_2d)

    return Response(content=buffer.getvalue(), media_type="application/octet-stream")


def _extract_parameter_group_and_key(key: str) -> tuple[str, str] | tuple[None, None]:
    key = key.removeprefix("LOG10_")
    if ":" not in key:
        # Assume all incoming keys are in format group:key for now
        return None, None

    param_group, param_key = key.split(":", maxsplit=1)
    return param_group, param_key


def data_for_parameter(ensemble: Ensemble, key: str) -> pd.DataFrame:
    group, _ = _extract_parameter_group_and_key(key)
    try:
        df = ensemble.load_scalars(group)
    except KeyError as e:
        logger.error(e)
        return pd.DataFrame()
    except Exception as ex:
        logger.exception(ex)
        raise HTTPException(status_code=500, detail="Internal server error") from ex

    dataframe = df.to_pandas().set_index("realization")
    dataframe.columns.name = None
    dataframe.index.name = "Realization"
    data = dataframe.sort_index(axis=1)
    if data.empty or key not in data:
        return pd.DataFrame()
    data = data[key].to_frame().dropna()
    data.columns = pd.Index([0])
    return data
